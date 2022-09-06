import logging

import nibabel as nib
import numpy as np
import os

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.utils import length
from scipy.ndimage import binary_dilation
from sklearn.cluster import KMeans

from scilpy.io.image import get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.segment.streamlines import filter_grid_roi, filter_grid_roi_both
from scilpy.tracking.tools import filter_streamlines_by_total_length_per_dim
from scilpy.tractanalysis.features import remove_loops_and_sharp_turns
from scilpy.tractanalysis.reproducibility_measures import \
    get_endpoints_density_map
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.filenames import split_name_with_nii


def compute_f1_score(overlap, overreach):
    """
    Compute the F1 score between overlap and overreach (they must be
    percentages).

    Parameters
    ------
    overlap: float, The overlap value.
    overreach: float, The overreach value
        (Version normalized over bundle area, not version normalized over gt).

    Returns
    -------
    f1_score: float, The f1 score.

    Ref: https://en.wikipedia.org/wiki/F1_score
    """
    # Recall = True positive / (True positive + False negative)
    #        = |B inter A| / |A|
    #        = overlap
    recall = overlap
    # Precision = True positive / (True positive + False positive)
    #        = |B inter A| / |B|
    #        = 1 - |B except A| / |B|
    #        = 1 - overreach
    precision = 1.0 - overreach
    f1_score = 2.0 * (precision * recall) / (precision + recall)
    return f1_score


def compute_f1_overlap_overreach(current_vb_voxels, gt_mask, dimensions):
    """
    Compute f1, OL and OR/ORn based on a ground truth mask.

    Parameters
    ------
    current_vb_voxels: 3D array
        The voxels touched by at least one streamlines for a given bundle.
    gt_mask: 3D array
        The ground truth mask.
    dimensions: np.array
        The nibabel dimensions of the data (3D).

    Returns
    -------
    f1: float
        The f1 score.
    tp_nb_voxels: int
        The TP (true positive) count in number of voxels.
    fp_nb_voxels: int
        The FP (false positive) count in number of voxels.
        Hint: Divide it by the ground truth count to get the overreach, or
        by the recovered bundle count to get the ORn (scores used in the
        ismrm2015 tractography challenge).
    fn_nb_voxels: int
        The number of voxels from the gt_mask that have not been recovered;
        corresponds to the FN count (false negative).
    overlap: float
        TP divided by the ground truth count (i.e. TP + FN), in percentage.
    overreach_pct_total: float
        The overreach, normalized by the recovered bundle's area. (Or 0 if
        no streamline have been recovered for this bundle).
    overreach_pct_gt: float
        The overreach, normalized by the ground truth area.
    """
    # True positive = |B inter A|
    tp_mask = gt_mask * current_vb_voxels
    tp_nb_voxels = np.count_nonzero(tp_mask)

    # False positive = |B except A|
    fp_mask = np.zeros(dimensions)
    fp_mask[np.where(
        (gt_mask == 0) & (current_vb_voxels >= 1))] = 1
    fp_nb_voxels = np.count_nonzero(fp_mask)

    # False negative = |A except B|
    fn_mask = np.zeros(dimensions)
    fn_mask[np.where(
        (gt_mask == 1) & (current_vb_voxels == 0))] = 1
    fn_nb_voxels = np.count_nonzero(fn_mask)

    gt_total_nb_voxels = tp_nb_voxels + fn_nb_voxels
    # Same as np.count_nonzero(gt_mask)

    nb_voxels_total = tp_nb_voxels + fp_nb_voxels
    # Same as np.count_nonzero(current_vb_voxels)

    # Overlap = |B inter A| / |A|
    overlap = tp_nb_voxels / gt_total_nb_voxels

    # Overreach: two versions are sometimes used.
    # |B except A| / |A| or |B except A| / |B|
    if nb_voxels_total == 0:
        overreach_pct_total = 0
    else:
        overreach_pct_total = fp_nb_voxels / nb_voxels_total
    overreach_pct_gt = fp_nb_voxels / gt_total_nb_voxels

    # f1 score (=dice)
    f1 = compute_f1_score(overlap, overreach_pct_total)

    return (f1, tp_nb_voxels, fp_nb_voxels, fn_nb_voxels,
            overlap, overreach_pct_gt, overreach_pct_total)


def get_binary_maps(streamlines, sft):
    """
    Extract a mask from a bundle.

    Parameters
    ----------
    streamlines: list
        List of streamlines.
    sft : StatefulTractogram
        Reference tractogram.

    Returns
    -------
    bundles_voxels: numpy.ndarray
        Mask representing the bundle volume.
    endpoints_voxels: numpy.ndarray
        Mask representing the bundle's endpoints.
    """
    dimensions = sft.dimensions
    if not len(streamlines):
        return np.zeros(dimensions), np.zeros(dimensions)
    elif len(streamlines) == 1:
        streamlines = [streamlines]
    tmp_sft = StatefulTractogram.from_sft(streamlines, sft)
    tmp_sft.to_vox()
    tmp_sft.to_corner()

    if len(tmp_sft) == 1:
        return np.zeros(dimensions), np.zeros(dimensions)

    bundles_voxels = compute_tract_counts_map(tmp_sft.streamlines,
                                              dimensions).astype(np.int16)

    endpoints_voxels = get_endpoints_density_map(tmp_sft.streamlines,
                                                 dimensions).astype(np.int16)

    bundles_voxels[bundles_voxels > 0] = 1
    endpoints_voxels[endpoints_voxels > 0] = 1

    return bundles_voxels, endpoints_voxels


def compute_masks(gt_files, parser, args):
    """
    Compute ground-truth masks. If the file is already a mask, load it.
    If it is a bundle, compute the mask.

    Parameters
    ----------
    gt_files: list
        List of either StatefulTractograms or niftis.
    parser: ArgumentParser
        Argument parser which handles the script's arguments.
    args: Namespace
        List of arguments passed to the script.

    Returns
    -------
    mask_1: numpy.ndarray
        "Head" of the mask.
    mask_2: numpy.ndarray
        "Tail" of the mask.
    """
    save_ref = args.reference

    gt_bundle_masks = []
    gt_bundle_inv_masks = []

    affine = None
    dimensions = None
    for gt_bundle in gt_files:
        if gt_bundle is not None:
            # Support ground truth as streamlines or masks
            # Will be converted to binary masks immediately
            _, ext = split_name_with_nii(gt_bundle)
            if ext in ['.gz', '.nii.gz']:
                gt_img = nib.load(gt_bundle)
                gt_mask = get_data_as_mask(gt_img)

                if affine is not None:
                    # compare affines.
                    # todO
                    logging.debug('Previous affine discarded. (todo)')
                affine = gt_img.affine
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
                    parser, args, gt_bundle, bbox_check=False)
                gt_sft.to_vox()
                gt_sft.to_corner()
                _affine, _dimensions, _, _ = gt_sft.space_attributes
                if affine is not None:
                    # compare affines.
                    # todO
                    logging.debug('Previous affine discarded. (todo)')
                affine = _affine
                dimensions = _dimensions
                gt_mask = compute_tract_counts_map(gt_sft.streamlines,
                                                   dimensions).astype(np.int16)
            gt_inv_mask = np.zeros(dimensions, dtype=np.int16)
            gt_inv_mask[gt_mask == 0] = 1
            gt_mask[gt_mask > 0] = 1
        else:
            gt_mask = None
            gt_inv_mask = None

        gt_bundle_masks.append(gt_mask)
        gt_bundle_inv_masks.append(gt_inv_mask)

    return gt_bundle_masks, gt_bundle_inv_masks, affine, dimensions


def split_heads_tails_kmeans(data):
    """
    Split a mask between head and tail with k means.

    Parameters
    ----------
    data: numpy.ndarray
        Mask to be split.

    Returns
    -------
    mask_1: numpy.ndarray
        "Head" of the mask.
    mask_2: numpy.ndarray
        "Tail" of the mask.
    """

    X = np.argwhere(data)
    k_means = KMeans(n_clusters=2).fit(X)
    mask_1 = np.zeros(data.shape)
    mask_2 = np.zeros(data.shape)

    mask_1[tuple(X[np.where(k_means.labels_ == 0)].T)] = 1
    mask_2[tuple(X[np.where(k_means.labels_ == 1)].T)] = 1

    return mask_1, mask_2


def extract_tails_heads_from_endpoints(gt_endpoints, out_dir):
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

    head, tail = split_heads_tails_kmeans(mask)

    basename = os.path.basename(
        split_name_with_nii(gt_endpoints)[0])
    tail_filename = os.path.join(
        out_dir, '{}_tail.nii.gz'.format(basename))
    head_filename = os.path.join(
        out_dir, '{}_head.nii.gz'.format(basename))
    nib.save(nib.Nifti1Image(head.astype(
        mask.dtype), affine), head_filename)
    nib.save(nib.Nifti1Image(tail.astype(
        mask.dtype), affine), tail_filename)

    return tail_filename, head_filename, affine, dimensions


def compute_endpoint_masks(roi_options, affine, dimensions, out_dir):
    """
    If endpoints without heads/tails are loaded, split them and continue
    normally after. Q/C of the output is important.

    Parameters
    ------
    roi_options: dict
        Keys are the bundle names. For each bundle, the value is itself a
        dictionary either key 'gt_endpoints' (the name of the file
        containing the bundle's endpoints), or both keys 'gt_tail' and
        'gt_head' (the names of the respetive files).
    affine: array
        A nibabel affine. Final masks must be compatible.
    dimensions: array
        A nibabel dimensions. Final masks must be compatible.
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
            tail, head, _affine, _dimensions = \
                extract_tails_heads_from_endpoints(
                    bundle_options['gt_endpoints'], out_dir)
            if affine is not None:
                # Compare affine
                # todo
                pass
            logging.debug('_affine discarded. (todo)')
            if dimensions is not None:
                # Compare dimensions
                # todo
                pass
            logging.debug("_dimensions discarded (todo)")
        else:
            tail = bundle_options['gt_tail']
            head = bundle_options['gt_head']

        tails.append(tail)
        heads.append(head)

    return tails, heads


def extract_vb_vs(
        sft, head_filename, tail_filename, limits_length, angle,
        orientation_length, abs_orientation_length, inclusion_inv_mask,
        dilate_endpoints):
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
    inclusion_inv_mask: np.ndarray or None
        Inverse mask of the bundle.
    dilate_endpoints: int or None
        If set, dilate the masks for n iterations.

    Returns
    -------
    tc_sft: StatefulTractogram
        SFT of true connections.
    wpc_sft: StatefulTractogram
        SFT of wrong-path-connections.
    fc_sft: StatefulTractogram
        SFT of false connections (streamlines that are too long).
    nc_streamlines: StatefulTractogram
        SFT of no connections (streamlines that loop)
    sft: StatefulTractogram
        SFT of remaining streamlines.
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
    if len(vs_ids) > 0 and inclusion_inv_mask is not None:
        tmp_sft = StatefulTractogram.from_sft(sft.streamlines[vs_ids], sft)
        _, out_of_mask_ids_from_vs = filter_grid_roi(
            tmp_sft, inclusion_inv_mask, 'any', False)
        out_of_mask_ids = vs_ids[out_of_mask_ids_from_vs]

        bundle_stats.update({"WPC_out_of_mask": len(out_of_mask_ids)})

        # Update ids
        wpc_ids.extend(out_of_mask_ids)
        vs_ids = np.setdiff1d(vs_ids, wpc_ids)

    # Remove invalid lengths
    if len(vs_ids) > 0 and limits_length is not None:
        min_len, max_len = limits_length

        # Bring streamlines to world coordinates so proper length
        # is calculated
        sft.to_rasmm()
        lengths = np.array(list(length(sft.streamlines[vs_ids])))
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
                sft[vs_ids], limits_x, limits_y,
                limits_z,
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


def extract_false_connections(sft, mask_1_filename, mask_2_filename,
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
