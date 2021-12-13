import nibabel as nib
import numpy as np
import os

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.utils import length
from nibabel.streamlines import ArraySequence
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


def extract_streamlines(mask_1, mask_2, sft):
    """
    Recognize streamlines between two masks

    Parameters
    ----------
    mask_1 : numpy.ndarray
        Mask containing one end of the bundle to be recognized.
    mask_2 : numpy.ndarray
        Mask containing the other end of the bundle to be recognized.
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.

    Returns
    -------
    extracted_sft: StatefulTractogram
        Tractogram containing the streamlines recognized.
    remaining_sft : StatefulTractogram
        Tractogram containing the streamlines not recognized.
    """

    extracted_sft, ids = filter_grid_roi_both(
        sft, mask_1, mask_2)
    remaining_ids = np.setdiff1d(range(len(sft.streamlines)), ids)
    remaining_sft = sft[list(remaining_ids)]

    return extracted_sft, remaining_sft


def get_binary_maps(streamlines, sft):
    """
    Extract a mask from a bundle

    Parameters
    ----------
    streamlines: list
        List of streamlines.
    dimensions: tuple of ints
        Dimensions of the mask.
    sft : StatefulTractogram
        Reference tractogram.
    invalid: bool
        If true, remove invalid streamlines from tractogram.

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


def compute_gt_masks(gt_bundles, parser, args):
    """
    Compute ground-truth masks. If the ground-truth is
    already a mask, load it. If the ground-truth is a
    bundle, compute the mask.

    Parameters
    ----------
    gt_bundles: list
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

    for gt_bundle in args.gt_bundles:
        # Support ground truth as streamlines or masks
        # Will be converted to binary masks immediately
        _, ext = split_name_with_nii(gt_bundle)
        if ext in ['.gz', '.nii.gz']:
            gt_img = nib.load(gt_bundle)
            gt_mask = get_data_as_mask(gt_img)
            affine = gt_img.affine
            dimensions = gt_mask.shape
        else:
            gt_sft = load_tractogram_with_reference(
                parser, args, gt_bundle, bbox_check=False)
            gt_sft.to_vox()
            gt_sft.to_corner()
            affine, dimensions, _, _ = gt_sft.space_attributes
            gt_mask = compute_tract_counts_map(gt_sft.streamlines,
                                               dimensions).astype(np.int16)
        gt_inv_mask = np.zeros(dimensions, dtype=np.int16)
        gt_inv_mask[gt_mask == 0] = 1
        gt_mask[gt_mask > 0] = 1
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
    gt_endpoints: list of str
        List of ground-truth mask filenames.

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

    tails = []
    heads = []
    for mask_filename in gt_endpoints:
        mask_img = nib.load(mask_filename)
        mask = get_data_as_mask(mask_img)
        affine = mask_img.affine
        dimensions = mask.shape

        head, tail = split_heads_tails_kmeans(mask)

        basename = os.path.basename(
            split_name_with_nii(mask_filename)[0])
        tail_filename = os.path.join(
            out_dir, '{}_tail.nii.gz'.format(basename))
        head_filename = os.path.join(
            out_dir, '{}_head.nii.gz'.format(basename))
        nib.save(nib.Nifti1Image(head.astype(
            mask.dtype), affine), head_filename)
        nib.save(nib.Nifti1Image(tail.astype(
            mask.dtype), affine), tail_filename)

        tails.append(tail_filename)
        heads.append(head_filename)

    return tails, heads, affine, dimensions


def extract_true_connections(
    sft, mask_1_filename, mask_2_filename, gt_config, length_dict,
    gt_bundle, gt_bundle_inv_mask, dilate_endpoints, wrong_path_as_separate
):
    """
    Extract true connections based on two regions from a tractogram.
    May extract false and no connections if the config is passed.

    Parameters
    ----------
    sft: StatefulTractogram
        Tractogram containing the streamlines to be extracted.
    mask_1_filename: str
        Filename of the "head" of the bundle.
    mask_2_filename: str
        Filename of the "tail" of the bundle.
    gt_config: dict or None
        Dictionary containing the bundle's parameters.
    length_dict: dict or None
        Dictionary containing the bundle's length parameters.
    gt_bundle: str
        Bundle's name.
    gt_bundle_inv_mask: np.ndarray
        Inverse mask of the bundle.
    dilate_endpoints: int or None
        If set, dilate the masks for n iterations.
    wrong_path_as_separate: bool
        If true, save the WPCs as separate from TCs.

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

    mask_1_img = nib.load(mask_1_filename)
    mask_2_img = nib.load(mask_2_filename)
    mask_1 = get_data_as_mask(mask_1_img)
    mask_2 = get_data_as_mask(mask_2_img)

    if dilate_endpoints:
        mask_1 = binary_dilation(mask_1, iterations=dilate_endpoints)
        mask_2 = binary_dilation(mask_2, iterations=dilate_endpoints)

    # TODO: Handle streamline IDs instead of streamlines
    tmp_sft, sft = extract_streamlines(mask_1, mask_2, sft)

    streamlines = tmp_sft.streamlines
    tc_streamlines = streamlines
    wpc_streamlines = []
    fc_streamlines = []
    nc_streamlines = []

    # Config file for each 'bundle'
    # Loops => no connection (nc) # TODO Is this legit ?
    # Length => false connection (fc) # TODO Is this legit ?
    if gt_config:
        min_len, max_len = \
            length_dict[gt_bundle]['length']

        # Bring streamlines to world coordinates so proper length
        # is calculated
        tmp_sft.to_rasmm()
        streamlines = tmp_sft.streamlines
        lengths = np.array(list(length(streamlines)))
        tmp_sft.to_vox()
        streamlines = tmp_sft.streamlines

        valid_min_length_mask = lengths > min_len
        valid_max_length_mask = lengths < max_len
        valid_length_mask = np.logical_and(valid_min_length_mask,
                                           valid_max_length_mask)
        streamlines = ArraySequence(streamlines)

        val_len_streamlines = streamlines[valid_length_mask]
        fc_streamlines = streamlines[~valid_length_mask]

        angle = length_dict[gt_bundle]['angle']
        tc_streamlines_ids = remove_loops_and_sharp_turns(
            val_len_streamlines, angle)

        loop_ids = np.setdiff1d(
            range(len(val_len_streamlines)), tc_streamlines_ids)

        loops = val_len_streamlines[list(loop_ids)]
        tc_streamlines = val_len_streamlines[list(tc_streamlines_ids)]

        if loops:
            nc_streamlines = loops

    # Streamlines getting out of the bundle mask can be considered
    # separately as wrong path connection (wpc)
    # TODO: Maybe only consider if they cross another GT bundle ?
    if wrong_path_as_separate:
        tmp_sft = StatefulTractogram.from_sft(tc_streamlines, sft)
        _, wp_ids = filter_grid_roi(
            tmp_sft, gt_bundle_inv_mask, 'any', False)
        wpc_streamlines = tmp_sft.streamlines[list(wp_ids)]
        tc_ids = np.setdiff1d(range(len(tmp_sft)), wp_ids)
        tc_streamlines = tmp_sft.streamlines[list(tc_ids)]

    tc_sft = StatefulTractogram.from_sft(tc_streamlines, sft)
    wpc_sft = StatefulTractogram.from_sft([], sft)
    fc_sft = StatefulTractogram.from_sft(fc_streamlines, sft)
    if wrong_path_as_separate and len(wpc_streamlines):
        wpc_sft = StatefulTractogram.from_sft(wpc_streamlines, sft)

    return tc_sft, wpc_sft, fc_sft, nc_streamlines, sft


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

    if len(sft.streamlines) > 0:
        tmp_sft, sft = extract_streamlines(mask_1, mask_2, sft)

        streamlines = tmp_sft.streamlines
        fc_streamlines = streamlines

        fc_sft = StatefulTractogram.from_sft(fc_streamlines, sft)
        return fc_sft, sft
    else:
        return sft, sft
