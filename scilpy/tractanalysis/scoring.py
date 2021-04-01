import nibabel as nib
import numpy as np
import os

from dipy.io.stateful_tractogram import StatefulTractogram
from sklearn.cluster import KMeans

from scilpy.image.operations import intersection
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.segment.streamlines import filter_grid_roi
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
    roi1 = nib.load(roi1_name).get_fdata().astype(np.int16)
    roi2 = nib.load(roi2_name).get_fdata().astype(np.int16)
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
            gt_mask = gt_img.get_fdata().astype(np.int16)
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
