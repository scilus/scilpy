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
from scilpy.tractanalysis.features import remove_loops_and_sharp_turns
from scilpy.tractanalysis.reproducibility_measures import \
    get_endpoints_density_map
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.filenames import split_name_with_nii


def get_binary_maps(streamlines, sft):
    """
    Extract a mask from a bundle

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
    Split a mask between head and tail with k means

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
    normally after. Q/C of the output is important

    Returns:
        tails, heads: lists of filenames with length the number of bundles.
    """
    tails = []
    heads = []
    for bundle_options in roi_options.values():
        if 'gt_endpoints' in bundle_options:
            tail, head, _affine, _dimensions = \
                extract_tails_heads_from_endpoints(
                    bundle_options['gt_endpoints'], out_dir)
            if affine is not None:
                # Compare affine
                # todo
                logging.debug('Affine discarded. (todo)')
        else:
            tail = bundle_options['gt_tail']
            head = bundle_options['gt_head']

        tails.append(tail)
        heads.append(head)

    return tails, heads


def make_sft_from_ids(ids, sft):
    if len(ids) > 0:
        streamlines = sft.streamlines[ids]
        data_per_streamline = sft.data_per_streamline[ids]
        data_per_point = sft.data_per_point[ids]
    else:
        streamlines = []
        data_per_streamline = None
        data_per_point = None

    new_sft = StatefulTractogram.from_sft(
        streamlines, sft,
        data_per_streamline=data_per_streamline,
        data_per_point=data_per_point)

    return new_sft


def extract_true_connections(
        sft, head_filename, tail_filename, limits_length, angle,
        bundle_prefix, inclusion_inv_mask, dilate_endpoints):
    """
    Extract true connections based on two regions from a tractogram.
    May extract false and no connections if the config is passed.

    Parameters
    ----------
    sft: StatefulTractogram
        Tractogram containing the streamlines to be extracted.
    head_filename: str
        Filename of the "head" of the bundle.
    tail_filename: str
        Filename of the "tail" of the bundle.
    limits_length: list
        Bundle's length parameters: [min max]
    angle: int
        Bundle's max angle.
    bundle_prefix: str
        Bundle's name.
    inclusion_inv_mask: np.ndarray
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

    _, tc_ids = filter_grid_roi_both(sft, mask_1, mask_2)

    wpc_ids = []
    bundle_stats = {"Bundle": bundle_prefix,
                    "Head": head_filename,
                    "Tail": tail_filename,
                    "Initial tc head to tail": len(tc_ids)}

    # Remove invalid lengths from tc
    if limits_length is not None:
        min_len, max_len = limits_length

        # Bring streamlines to world coordinates so proper length
        # is calculated
        sft.to_rasmm()
        lengths = np.array(list(length(sft.streamlines[tc_ids])))
        sft.to_vox()

        # Compute valid lengths
        valid_length_ids_mask_from_tc = np.logical_and(lengths > min_len,
                                                       lengths < max_len)

        bundle_stats.update({
            "WPC_invalid_length": sum(~valid_length_ids_mask_from_tc)})

        # Update ids
        wpc_ids.extend(tc_ids[~valid_length_ids_mask_from_tc])
        tc_ids = tc_ids[valid_length_ids_mask_from_tc]

    # Remove loops from tc
    if angle is not None:
        # Compute valid angles
        valid_angle_ids_from_tc = remove_loops_and_sharp_turns(
            sft.streamlines[tc_ids], angle)

        # Update ids
        valid_angle_ids = tc_ids[valid_angle_ids_from_tc]
        invalid_angle_ids = np.setdiff1d(tc_ids, valid_angle_ids)

        bundle_stats.update({"WPC_invalid_length": len(invalid_angle_ids)})

        wpc_ids.extend(invalid_angle_ids)
        tc_ids = valid_angle_ids

    # Streamlines getting out of the bundle mask can be considered
    # separately as wrong path connection (wpc)
    if inclusion_inv_mask is not None:

        tmp_sft = StatefulTractogram.from_sft(sft.streamlines[tc_ids], sft)
        _, out_of_mask_ids_from_tc = filter_grid_roi(
            tmp_sft, inclusion_inv_mask, 'any', False)
        out_of_mask_ids = tc_ids[out_of_mask_ids_from_tc]

        bundle_stats.update({"WPC_out_of_mask": len(out_of_mask_ids)})

        # Update ids
        wpc_ids.extend(out_of_mask_ids)
        tc_ids = np.setdiff1d(tc_ids, wpc_ids)

        bundle_stats.update({"TC": len(tc_ids)})

    return list(tc_ids), list(wpc_ids), bundle_stats


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

    fc_sft = make_sft_from_ids(fc_ids, sft)
    return fc_sft, fc_ids
