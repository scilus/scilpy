# -*- coding: utf-8 -*-

import itertools
import logging

import nibabel as nib
import numpy as np
import os
from scipy.ndimage import binary_dilation

from dipy.io.streamline import save_tractogram
from dipy.tracking.utils import length as compute_length

from scilpy.io.image import get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.segment.streamlines import filter_grid_roi, filter_grid_roi_both
from scilpy.tractograms.streamline_operations import \
    remove_loops_and_sharp_turns
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map

from scilpy.tractograms.streamline_and_mask_operations import \
    split_mask_blobs_kmeans
from scilpy.tractograms.streamline_operations import \
    filter_streamlines_by_total_length_per_dim
from scilpy.utils.filenames import split_name_with_nii


def _extract_prefix(filename):
    prefix = os.path.basename(filename)
    prefix, _ = split_name_with_nii(prefix)

    return prefix


def compute_masks_from_bundles(gt_files, parser, args, inverse_mask=False):
    """
    Compute ground-truth masks. If the file is already a mask, load it.
    If it is a bundle, compute the mask. If the filename is None, appends None
    to the lists of masks. Compatibility between files should already be
    verified.

    Parameters
    ----------
    gt_files: list
        List of either StatefulTractograms or niftis.
    parser: ArgumentParser
        Argument parser which handles the script's arguments. Used to print
        parser errors, if any.
    args: Namespace
        List of arguments passed to the script. Used for its 'ref' and
        'bbox_check' arguments.
    inverse_mask: bool
        If true, returns the list of inversed masks instead.

    Returns
    -------
    mask: list[numpy.ndarray]
        The loaded masks.
    """
    save_ref = args.reference

    gt_bundle_masks = []

    for gt_bundle in gt_files:
        if gt_bundle is not None:
            # Support ground truth as streamlines or masks
            # Will be converted to binary masks immediately
            _, ext = split_name_with_nii(gt_bundle)
            if ext in ['.gz', '.nii.gz']:
                gt_img = nib.load(gt_bundle)
                gt_mask = get_data_as_mask(gt_img)
                dimensions = gt_mask.shape
            else:
                # Cheating ref because it may send a lot of warning if loading
                # many trk with ref (reference was maybe added only for some
                # of these files)
                if ext == '.trk':
                    args.reference = None
                else:
                    args.reference = save_ref
                gt_sft = load_tractogram_with_reference(
                    parser, args, gt_bundle)
                gt_sft.to_vox()
                gt_sft.to_corner()
                _, dimensions, _, _ = gt_sft.space_attributes
                gt_mask = compute_tract_counts_map(gt_sft.streamlines,
                                                   dimensions).astype(np.int16)
            gt_mask[gt_mask > 0] = 1

            if inverse_mask:
                gt_inv_mask = np.zeros(dimensions, dtype=np.int16)
                gt_inv_mask[gt_mask == 0] = 1
                gt_mask = gt_inv_mask
        else:
            gt_mask = None

        gt_bundle_masks.append(gt_mask)

    return gt_bundle_masks


def _extract_and_save_tails_heads_from_endpoints(gt_endpoints, out_dir):
    """
    Extract two masks from a single mask containing two regions.

    Parameters
    ----------
    gt_endpoints: str
        Ground-truth mask filename.
    out_dir: str
        Path where to save the heads and tails.

    Returns
    -------
    tails: list
        List of tail filenames.
    heads: list
        List of head filenames.
    affine: numpy.ndarray
        Affine of mask image.
    dimensions: tuple of int
        Dimensions of the mask image.
    """
    mask_img = nib.load(gt_endpoints)
    mask = get_data_as_mask(mask_img)
    affine = mask_img.affine
    dimensions = mask.shape

    head, tail = split_mask_blobs_kmeans(mask, nb_clusters=2)

    basename = os.path.basename(split_name_with_nii(gt_endpoints)[0])
    tail_filename = os.path.join(out_dir, '{}_tail.nii.gz'.format(basename))
    head_filename = os.path.join(out_dir, '{}_head.nii.gz'.format(basename))
    nib.save(nib.Nifti1Image(head.astype(mask.dtype), affine), head_filename)
    nib.save(nib.Nifti1Image(tail.astype(mask.dtype), affine), tail_filename)

    return tail_filename, head_filename, affine, dimensions


def compute_endpoint_masks(roi_options, out_dir):
    """
    If endpoints without heads/tails are loaded, split them and continue
    normally after. Q/C of the output is important. Compatibility between files
    should be already verified.

    Parameters
    ------
    roi_options: dict
        Keys are the bundle names. For each bundle, the value is itself a
        dictionary either key 'gt_endpoints' (the name of the file
        containing the bundle's endpoints), or both keys 'gt_tail' and
        'gt_head' (the names of the respetive files).
    out_dir: str
        Where to save the heads and tails.

    Returns
    -------
    tails, heads: lists of filenames with length the number of bundles.
    """
    tails = []
    heads = []
    for bundle_options in roi_options:
        if 'gt_endpoints' in bundle_options:
            tail, head, _, _ = _extract_and_save_tails_heads_from_endpoints(
                bundle_options['gt_endpoints'], out_dir)
        else:
            tail = bundle_options['gt_tail']
            head = bundle_options['gt_head']

        tails.append(tail)
        heads.append(head)

    return tails, heads


def _extract_vb_and_wpc_all_bundles(
        gt_tails, gt_heads, sft, bundle_names, lengths, angles,
        orientation_lengths, abs_orientation_lengths, inv_all_masks,
        any_masks, args):
    """
    Loop on every ground truth bundles and extract VS and WPC.

    VS:
       1) Connect the head and tail
       2) Are completely included in the all_mask (if any)
       3) Have acceptable angle, length and length per orientation.
       4) Reach the any_mask (if any)
     +
    WPC connections:
       1) connect the head and tail but criteria 2 and 3 are not respected

    Returns
    -------
    vb_sft_list: list
        List of StatefulTractograms of VS
    wpc_sft_list: list
        List of StatefulTractograms of WPC if args.save_wpc_separately), else
        None.
    all_vs_wpc_ids: list
        List of list of all VS + WPC streamlines detected.
    bundle_stats_dict: dict
        Dictionnary of the processing information for each bundle.

    Saves
    -----
    - Each duplicate in segmented_conflicts/duplicates_*_*.trk
    """
    nb_bundles = len(bundle_names)

    vb_sft_list = []
    vs_ids_list = []
    wpc_ids_list = []
    bundles_stats = []

    remaining_ids = np.arange(len(sft))  # For args.unique management.

    # 1. Extract VB and WPC.
    for i in range(nb_bundles):
        head_filename = gt_heads[i]
        tail_filename = gt_tails[i]

        vs_ids, wpc_ids, bundle_stats = \
            _extract_vb_one_bundle(
                sft[remaining_ids], head_filename, tail_filename, lengths[i],
                angles[i], orientation_lengths[i], abs_orientation_lengths[i],
                inv_all_masks[i], any_masks[i], args.dilate_endpoints)

        if args.unique:
            # Assign actual ids, not from subset
            vs_ids = remaining_ids[vs_ids]
            wpc_ids = remaining_ids[wpc_ids]
            # Update remaining_ids based on valid streamlines only
            remaining_ids = np.setdiff1d(remaining_ids, vs_ids,
                                         assume_unique=True)

        # Append info
        vb_sft = sft[vs_ids]
        vb_sft_list.append(vb_sft)
        vs_ids_list.append(vs_ids)
        wpc_ids_list.append(wpc_ids)
        bundles_stats.append(bundle_stats)

        logging.info("Bundle {}: nb VS = {}"
                     .format(bundle_names[i], bundle_stats["VS"]))
    all_gt_ids = list(itertools.chain(*vs_ids_list))

    # 2. Remove duplicate WPC and then save.
    if args.save_wpc_separately:
        if args.remove_wpc_belonging_to_another_bundle or args.unique:
            for i in range(nb_bundles):
                new_wpc_ids = np.setdiff1d(wpc_ids_list[i], all_gt_ids)
                nb_rejected = len(wpc_ids_list[i]) - len(new_wpc_ids)
                bundles_stats[i].update(
                    {"Belonging to another bundle": nb_rejected})
                wpc_ids_list[i] = new_wpc_ids
                bundles_stats[i].update({"Cleaned WPC": len(new_wpc_ids)})

        wpc_sft_list = []
        for i in range(nb_bundles):
            logging.info("Bundle {}: nb WPC = {}"
                         .format(bundle_names[i], len(wpc_ids_list[i])))
            wpc_ids = wpc_ids_list[i]
            if len(wpc_ids) == 0:
                wpc_sft = None
            else:
                wpc_sft = sft[wpc_ids]
            wpc_sft_list.append(wpc_sft)
    else:
        # Remove WPCs to be included as IS in the future
        wpc_ids_list = [[] for _ in range(nb_bundles)]
        wpc_sft_list = None

    # 3. If not args.unique, tell users if there were duplicates. Save
    # duplicates separately in segmented_conflicts/duplicates_*_*.trk.
    if not args.unique:
        for i in range(nb_bundles):
            for j in range(i + 1, nb_bundles):
                duplicate_ids = np.intersect1d(vs_ids_list[i], vs_ids_list[j])
                if len(duplicate_ids) > 0:
                    logging.warning(
                        "{} streamlines belong to true connections of both "
                        "bundles {} and {}.\n"
                        "Please verify your criteria!"
                        .format(len(duplicate_ids), bundle_names[i],
                                bundle_names[j]))

                    # Duplicates directory only created if at least one
                    # duplicate is found.
                    path_duplicates = os.path.join(args.out_dir,
                                                   'segmented_conflicts')
                    if not os.path.isdir(path_duplicates):
                        os.makedirs(path_duplicates)

                    save_tractogram(sft[duplicate_ids], os.path.join(
                        path_duplicates, 'duplicates_' + bundle_names[i] +
                        '_' + bundle_names[j] + '.trk'))

    # 4. Save bundle stats.
    bundle_stats_dict = {}
    for i in range(len(bundle_names)):
        bundle_stats_dict.update({bundle_names[i]: bundles_stats[i]})

    all_vs_ids = np.unique(list(itertools.chain(*vs_ids_list)))
    all_wpc_ids = np.unique(list(itertools.chain(*wpc_ids_list)))
    all_vs_wpc_ids = np.concatenate((all_vs_ids, all_wpc_ids))

    return vb_sft_list, wpc_sft_list, all_vs_wpc_ids, bundle_stats_dict


def _extract_vb_one_bundle(
        sft, head_filename, tail_filename, limits_length, angle,
        orientation_length, abs_orientation_length, inv_all_mask,
        any_mask, dilate_endpoints):
    """
    Extract valid bundle (and valid streamline ids) from a tractogram, based
    on two regions of interest for the endpoints, one region of interest for
    the inclusion of streamlines, and maximum length, maximum angle,
    maximum length per orientation.

    Parameters
    ----------
    sft: StatefulTractogram
        Tractogram containing the streamlines to be extracted.
    head_filename: str
        Filename of the "head" of the bundle.
    tail_filename: str
        Filename of the "tail" of the bundle.
    limits_length: list or None
        Bundle's length parameters: [min max].
    angle: int or None
        Bundle's max angle.
    orientation_length: list or None
        Bundle's length parameters in each direction:
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    abs_orientation_length: idem, computed in absolute values.
    inv_all_mask: np.ndarray or None
        Inverse ALL mask for this bundle: no point must be outside the mask.
    any_mask: np.ndarray or None
        ANY mask for this bundle.
        Streamlines must pass through this mask (touch it) to be included
        in the bundle.
    dilate_endpoints: int or None
        If set, dilate the masks for n iterations.

    Returns
    -------
    vs_ids: list
        List of ids of valid streamlines
    wpc_ids: list
        List of ids of wrong-path connections
    bundle_stats: dict
        Dictionary of recognized streamlines statistics
    """
    mask_1_img = nib.load(head_filename)
    mask_2_img = nib.load(tail_filename)
    mask_1 = get_data_as_mask(mask_1_img)
    mask_2 = get_data_as_mask(mask_2_img)

    if dilate_endpoints:
        mask_1 = binary_dilation(mask_1, iterations=dilate_endpoints)
        mask_2 = binary_dilation(mask_2, iterations=dilate_endpoints)

    _, vs_ids = filter_grid_roi_both(sft, mask_1, mask_2)

    wpc_ids = []
    bundle_stats = {"Initial count head to tail": len(vs_ids)}

    # Remove out of inclusion mask (limits_mask)
    if len(vs_ids) > 0 and inv_all_mask is not None:
        tmp_sft = sft[vs_ids]
        _, out_of_mask_ids_from_vs = filter_grid_roi(
            tmp_sft, inv_all_mask, 'any', False)
        out_of_mask_ids = vs_ids[out_of_mask_ids_from_vs]

        bundle_stats.update({"WPC_out_of_mask": len(out_of_mask_ids)})

        # Update ids
        wpc_ids.extend(out_of_mask_ids)
        vs_ids = np.setdiff1d(vs_ids, wpc_ids)

    # Remove streamlines not passing through any_mask
    if len(vs_ids) > 0 and any_mask is not None:
        tmp_sft = sft[vs_ids]
        _, in_mask_ids_from_vs = filter_grid_roi(
            tmp_sft, any_mask, 'any', False)
        in_mask_ids = vs_ids[in_mask_ids_from_vs]

        out_of_mask_ids = np.setdiff1d(vs_ids, in_mask_ids)
        bundle_stats.update({"WPC_not_reaching_the_ANY_mask":
                             len(out_of_mask_ids)})

        # Update ids
        wpc_ids.extend(out_of_mask_ids)
        vs_ids = in_mask_ids

    # Remove invalid lengths
    if len(vs_ids) > 0 and limits_length is not None:
        min_len, max_len = limits_length

        # Bring streamlines to world coordinates so proper length
        # is calculated
        sft.to_rasmm()
        lengths = np.array(list(compute_length(sft.streamlines[vs_ids])))
        sft.to_vox()

        # Compute valid lengths
        valid_length_ids_mask_from_vs = np.logical_and(lengths > min_len,
                                                       lengths < max_len)

        bundle_stats.update({
            "WPC_invalid_length": int(sum(~valid_length_ids_mask_from_vs))})

        # Update ids
        wpc_ids.extend(vs_ids[~valid_length_ids_mask_from_vs])
        vs_ids = vs_ids[valid_length_ids_mask_from_vs]

    # Remove invalid lengths per orientation
    if len(vs_ids) > 0 and orientation_length is not None:
        # Compute valid lengths
        limits_x, limits_y, limits_z = orientation_length

        _, valid_orientation_ids_from_vs, _ = \
            filter_streamlines_by_total_length_per_dim(
                sft[vs_ids], limits_x, limits_y, limits_z,
                use_abs=False, save_rejected=False)

        # Update ids
        valid_orientation_ids = vs_ids[valid_orientation_ids_from_vs]
        invalid_orientation_ids = np.setdiff1d(vs_ids, valid_orientation_ids)

        bundle_stats.update({
            "WPC_invalid_orientation": len(invalid_orientation_ids)})

        wpc_ids.extend(invalid_orientation_ids)
        vs_ids = valid_orientation_ids

    # Idem in abs
    if len(vs_ids) > 0 and abs_orientation_length is not None:
        # Compute valid lengths
        limits_x, limits_y, limits_z = abs_orientation_length

        _, valid_orientation_ids_from_vs, _ = \
            filter_streamlines_by_total_length_per_dim(
                sft[vs_ids], limits_x, limits_y, limits_z,
                use_abs=True, save_rejected=False)

        # Update ids
        valid_orientation_ids = vs_ids[valid_orientation_ids_from_vs]
        invalid_orientation_ids = np.setdiff1d(vs_ids,
                                               valid_orientation_ids)

        bundle_stats.update({
            "WPC_invalid_orientation_abs": len(invalid_orientation_ids)})

        wpc_ids.extend(invalid_orientation_ids)
        vs_ids = valid_orientation_ids

    # Remove loops from tc
    if len(vs_ids) > 0 and angle is not None:
        # Compute valid angles
        valid_angle_ids_from_vs = remove_loops_and_sharp_turns(
            sft.streamlines[vs_ids], angle)

        # Update ids
        valid_angle_ids = vs_ids[valid_angle_ids_from_vs]
        invalid_angle_ids = np.setdiff1d(vs_ids, valid_angle_ids)

        bundle_stats.update({"WPC_invalid_length": len(invalid_angle_ids)})

        wpc_ids.extend(invalid_angle_ids)
        vs_ids = valid_angle_ids

    bundle_stats.update({"VS": len(vs_ids)})

    return list(vs_ids), list(wpc_ids), bundle_stats


def _extract_ib_one_bundle(sft, mask_1_filename, mask_2_filename,
                           dilate_endpoints):
    """
    Extract false connections based on two regions from a tractogram.

    Parameters
    ----------
    sft: StatefulTractogram
        Tractogram containing the streamlines to be extracted.
    mask_1_filename: str
        Filename of the "head" of the bundle.
    mask_2_filename: str
        Filename of the "tail" of the bundle.
    dilate_endpoints: int or None
        If set, dilate the masks for n iterations.

    Returns
    -------
    fc_sft: StatefulTractogram
        SFT of false connections.
    sft: StatefulTractogram
        SFT of remaining streamlines.
    """

    mask_1_img = nib.load(mask_1_filename)
    mask_2_img = nib.load(mask_2_filename)
    mask_1 = get_data_as_mask(mask_1_img)
    mask_2 = get_data_as_mask(mask_2_img)

    if dilate_endpoints:
        mask_1 = binary_dilation(mask_1, iterations=dilate_endpoints)
        mask_2 = binary_dilation(mask_2, iterations=dilate_endpoints)

    _, fc_ids = filter_grid_roi_both(sft, mask_1, mask_2)

    fc_sft = sft[fc_ids]
    return fc_sft, fc_ids


def _extract_ib_all_bundles(comb_filename, sft, args):
    """
    Loop on every bundle and compute false connections, defined as connections
    between ROIs pairs that do not form gt bundles.

    (Goes through all the possible combinations of endpoints masks)
    """
    ib_sft_list = []
    ic_ids_list = []
    ib_bundle_names = []

    all_ids = np.arange(len(sft))
    for i, roi in enumerate(comb_filename):
        roi1_filename, roi2_filename = roi

        # Automatically generate filename for Q/C
        prefix_1 = _extract_prefix(roi1_filename)
        prefix_2 = _extract_prefix(roi2_filename)

        ib_sft, ic_ids = _extract_ib_one_bundle(
            sft[all_ids], roi1_filename, roi2_filename, args.dilate_endpoints)

        if args.unique:
            ic_ids = all_ids[ic_ids]
            all_ids = np.setdiff1d(all_ids, ic_ids, assume_unique=True)

        if len(ib_sft.streamlines) > 0:
            logging.info("IB: Recognized {} streamlines between {} and {}"
                         .format(len(ib_sft.streamlines), prefix_1, prefix_2))

            ib_sft_list.append(ib_sft)
            ic_ids_list.append(ic_ids)
            ib_bundle_names.append(prefix_1 + '_' + prefix_2)

    # Duplicates?
    if not args.unique:
        nb_pairs = len(ic_ids_list)
        for i in range(nb_pairs):
            for j in range(i + 1, nb_pairs):
                duplicate_ids = np.intersect1d(ic_ids_list[i], ic_ids_list[j])
                if len(duplicate_ids) > 0:
                    logging.warning(
                        "{} streamlines are scored twice as invalid "
                        "connections\n (between pair {}\n and between pair "
                        "{}). You probably have overlapping ROIs!"
                        .format(len(duplicate_ids), comb_filename[i],
                                comb_filename[j]))

    return ib_sft_list, ic_ids_list, ib_bundle_names


def segment_tractogram_from_roi(
        sft, gt_tails, gt_heads, bundle_names, bundle_lengths, angles,
        orientation_lengths, abs_orientation_lengths, inv_all_masks, any_masks,
        list_rois, args):
    """
    Segments valid bundles (VB). Based on args:
    - args.compute_ic: computes invalid bundles (IB)
    - args.save_wpc_separately: compute WPC

    Returns
    -------
    vb_sft_list: list
        The list of valid bundles discovered. These files are also saved
        in segmented_VB/\\*_VS.trk.
    wpc_sft_list: list
        The list of wrong path connections: streamlines connecting the right
        endpoint regions but not included in the ALL mask.
        ** This is only computed if args.save_wpc_separately. Else, this is
        None.
    ib_sft_list: list
        The list of invalid bundles: streamlines connecting regions that should
        not be connected.
        ** This is only computed if args.compute_ic. Else, this is None.
    nc_sft_list: list
        The list of rejected streamlines that were not included in any IB.
    ib_names: list
        The list of names for invalid bundles (IB). They are created from the
        combinations of ROIs used for IB computations.
    bundle_stats: dict
        Dictionnary of the processing information for each VB bundle.
    """
    sft.to_vox()

    # VS
    logging.info("Extracting valid bundles (and wpc, if any)")
    vb_sft_list, wpc_sft_list, detected_vs_wpc_ids, bundle_stats = \
        _extract_vb_and_wpc_all_bundles(
            gt_tails, gt_heads, sft, bundle_names, bundle_lengths,
            angles, orientation_lengths, abs_orientation_lengths,
            inv_all_masks, any_masks, args)

    remaining_ids = np.arange(0, len(sft))
    if args.unique:
        remaining_ids = np.setdiff1d(remaining_ids, detected_vs_wpc_ids)

    # IC
    if args.compute_ic and len(remaining_ids) > 0:
        logging.info("Extracting invalid bundles")

        # Keep all possible combinations
        list_rois = sorted(list_rois)
        comb_filename = list(itertools.combinations(list_rois, r=2))

        # Remove the true connections from all combinations, leaving only
        # false connections
        vb_roi_filenames = list(zip(gt_tails, gt_heads))
        for vb_roi_pair in vb_roi_filenames:
            vb_roi_pair = tuple(sorted(vb_roi_pair))
            comb_filename.remove(vb_roi_pair)
        ib_sft_list, ic_ids_list, ib_names = _extract_ib_all_bundles(
            comb_filename, sft[remaining_ids], args)
        if args.unique and len(ic_ids_list) > 0:
            for i in range(len(ic_ids_list)):
                # Assign actual ids
                ic_ids_list[i] = remaining_ids[ic_ids_list[i]]
            detected_vs_wpc_ids = np.concatenate(ic_ids_list)
            remaining_ids = np.setdiff1d(remaining_ids, detected_vs_wpc_ids)
    else:
        ic_ids_list = []
        ib_sft_list = []
        ib_names = []

    all_ic_ids = np.unique(list(itertools.chain(*ic_ids_list)))

    # NC
    # = ids that are not VS, not wpc (if asked) and not IC (if asked).
    all_nc_ids = remaining_ids
    if not args.unique:
        all_nc_ids = np.setdiff1d(all_nc_ids, detected_vs_wpc_ids)
        all_nc_ids = np.setdiff1d(all_nc_ids, all_ic_ids)

    if args.compute_ic:
        logging.info("The remaining {} / {} streamlines will be scored as NC."
                     .format(len(all_nc_ids), len(sft)))
        filename = "NC.trk"
    else:
        logging.info("The remaining {} / {} streamlines will be scored as IS."
                     .format(len(all_nc_ids), len(sft)))
        filename = "IS.trk"

    nc_sft = sft[all_nc_ids]
    if len(nc_sft) > 0 or not args.no_empty:
        save_tractogram(nc_sft, os.path.join(
            args.out_dir, filename), bbox_valid_check=args.bbox_check)

    return (vb_sft_list, wpc_sft_list, ib_sft_list, nc_sft, ib_names,
            bundle_stats)
