import nibabel as nib
import numpy as np
import os

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.utils import length
from nibabel.streamlines import ArraySequence
from scipy.ndimage import binary_dilation
from sklearn.cluster import KMeans

from scilpy.image.operations import intersection
from scilpy.io.image import get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.segment.streamlines import filter_grid_roi
from scilpy.tractanalysis.features import remove_loops_and_sharp_turns
from scilpy.tractanalysis.reproducibility_measures import \
    get_endpoints_density_map
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.filenames import split_name_with_nii
from scilpy.utils.streamlines import \
    (difference, perform_streamlines_operation)


def extract_streamlines(mask_1, mask_2, sft):

    tmp_sft, mask_1_ids = filter_grid_roi(
        sft, mask_1, 'either_end', False)
    extracted_sft, mask_2_ids = filter_grid_roi(
        tmp_sft, mask_2, 'either_end', False)
    remaining_ids = np.setdiff1d(range(len(sft.streamlines)),
                                 mask_1_ids[list(mask_2_ids)])
    remaining_sft = sft[list(remaining_ids)]

    return extracted_sft, remaining_sft


def get_binary_maps(streamlines, dimensions, sft, invalid):
    if not len(streamlines):
        return np.zeros(dimensions), np.zeros(dimensions)
    elif len(streamlines) == 1:
        streamlines = [streamlines]
    tmp_sft = StatefulTractogram.from_sft(streamlines, sft)
    tmp_sft.to_vox()
    tmp_sft.to_corner()
    if invalid:
        tmp_sft.remove_invalid_streamlines()

    if len(tmp_sft) == 1:
        return np.zeros(dimensions), np.zeros(dimensions)

    bundles_voxels = compute_tract_counts_map(tmp_sft.streamlines,
                                              dimensions).astype(np.int16)

    endpoints_voxels = get_endpoints_density_map(tmp_sft.streamlines,
                                                 dimensions).astype(np.int16)

    bundles_voxels[bundles_voxels > 0] = 1
    endpoints_voxels[endpoints_voxels > 0] = 1

    return bundles_voxels, endpoints_voxels


def identify_overlapping_roi(mask_1, mask_2):
    overlapping_roi = []

    if not os.path.isfile(mask_1):
        raise ValueError('Input file {} does not exist.'.format(mask_1))
    roi_1 = nib.load(mask_1)
    roi1 = roi_1.get_fdata(dtype=np.float64)

    if not os.path.isfile(mask_2):
        raise ValueError('Input file {} does not exist.'.format(mask_2))
    roi_2 = nib.load(mask_2)
    roi2 = roi_2.get_fdata(dtype=np.float64)

    rois = [roi1, roi2]
    overlap = intersection(rois, roi_1)
    nb_voxels = np.count_nonzero(overlap)

    if nb_voxels > 0:
        overlapping_roi.append((mask_1, mask_2))
        overlapping_roi.append((mask_2, mask_1))

    return overlapping_roi


def remove_duplicate_streamlines(sft, fc_streamlines, roi1_name, roi2_name):
    roi1 = get_data_as_mask(roi1_name)
    roi2 = get_data_as_mask(roi2_name)
    tmp_sft, _ = filter_grid_roi(sft, roi1, 'either_end', False)
    tmp_sft, _ = filter_grid_roi(tmp_sft, roi2, 'either_end', False)
    duplicate_streamlines = tmp_sft.streamlines
    fc_streamlines, _ = perform_streamlines_operation(
        difference, [fc_streamlines, duplicate_streamlines], precision=0)
    return fc_streamlines


def split_heads_tails_kmeans(data):
    X = np.argwhere(data)
    k_means = KMeans(n_clusters=2).fit(X)
    mask_1 = np.zeros(data.shape)
    mask_2 = np.zeros(data.shape)

    mask_1[tuple(X[np.where(k_means.labels_ == 0)].T)] = 1
    mask_2[tuple(X[np.where(k_means.labels_ == 1)].T)] = 1

    return mask_1, mask_2


def compute_gt_masks(gt_bundles, parser, args):
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
            if args.remove_invalid:
                gt_sft.remove_invalid_streamlines()
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


def extract_tails_heads_from_endpoints(gt_endpoints, out_dir):
    tails = []
    heads = []
    for mask_filename in gt_endpoints:
        mask_img = nib.load(mask_filename)
        mask = mask_img.get_fdata().astype(np.int16)
        affine = mask_img.affine
        dimensions = mask.shape

        head, tail = split_heads_tails_kmeans(mask)

        basename = os.path.basename(
            split_name_with_nii(mask_filename)[0])
        tail_filename = os.path.join(
            out_dir, '{}_tail.nii.gz'.format(basename))
        head_filename = os.path.join(
            out_dir, '{}_head.nii.gz'.format(basename))
        nib.save(nib.Nifti1Image(head, affine), head_filename)
        nib.save(nib.Nifti1Image(tail, affine), tail_filename)

        tails.append(tail_filename)
        heads.append(head_filename)

    return tails, heads, affine, dimensions


def extract_true_connections(
    sft, mask_1_filename, mask_2_filename, gt_config, length_dict,
    gt_bundle, gt_bundle_inv_mask, dilate_endpoints, wrong_path_as_separate
):
    mask_1_img = nib.load(mask_1_filename)
    mask_2_img = nib.load(mask_2_filename)
    mask_1 = get_data_as_mask(mask_1_img)
    mask_2 = get_data_as_mask(mask_2_img)

    if dilate_endpoints:
        mask_1 = binary_dilation(mask_1, iterations=dilate_endpoints)
        mask_2 = binary_dilation(mask_2, iterations=dilate_endpoints)

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

        lengths = np.array(list(length(streamlines)))
        valid_min_length_mask = lengths > min_len
        valid_max_length_mask = lengths < max_len
        valid_length_mask = np.logical_and(valid_min_length_mask,
                                           valid_max_length_mask)
        streamlines = ArraySequence(streamlines)

        val_len_streamlines = streamlines[valid_length_mask]
        fc_streamlines = streamlines[~valid_length_mask]

        angle = length_dict[gt_bundle]['angle']
        tc_streamlines, loops = remove_loops_and_sharp_turns(
            val_len_streamlines, angle)

        if loops:
            nc_streamlines = loops

    # Streamlines getting out of the bundle mask can be considered
    # separately as wrong path connection (wpc)
    # TODO: Can they ? Maybe only consider if they cross another
    # GT bundle ?
    if wrong_path_as_separate:
        tmp_sft = StatefulTractogram.from_sft(tc_streamlines, sft)
        wpc_stf, _ = filter_grid_roi(
            tmp_sft, gt_bundle_inv_mask, 'any', False)
        wpc_streamlines = wpc_stf.streamlines
        tc_streamlines, _ = perform_streamlines_operation(
            difference, [tc_streamlines, wpc_streamlines], precision=0)

    tc_sft = StatefulTractogram.from_sft(tc_streamlines, sft)
    wpc_sft = StatefulTractogram.from_sft([], sft)
    fc_sft = StatefulTractogram.from_sft(fc_streamlines, sft)
    if wrong_path_as_separate:
        wpc_sft = StatefulTractogram.from_sft(wpc_streamlines, sft)

    return tc_sft, wpc_sft, fc_sft, nc_streamlines, sft


def extract_false_connections(
    sft, mask_1_filename, mask_2_filename, dilate_endpoints
):
    mask_1_img = nib.load(mask_1_filename)
    mask_2_img = nib.load(mask_2_filename)
    mask_1 = get_data_as_mask(mask_1_img)
    mask_2 = get_data_as_mask(mask_2_img)

    if dilate_endpoints:
        mask_1 = binary_dilation(mask_1, iterations=dilate_endpoints)
        mask_2 = binary_dilation(mask_2, iterations=dilate_endpoints)

    tmp_sft, sft = extract_streamlines(mask_1, mask_2, sft)

    streamlines = tmp_sft.streamlines
    fc_streamlines = streamlines

    fc_sft = StatefulTractogram.from_sft(fc_streamlines, sft)
    return fc_sft, sft
