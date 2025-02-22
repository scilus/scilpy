# -*- coding: utf-8 -*-

import os

import nibabel as nib
import numpy as np
from dipy.io.streamline import load_tractogram

from scilpy import SCILPY_HOME
from scilpy.image.utils import split_mask_blobs_kmeans
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.tractograms.streamline_and_mask_operations import (
    _intersects_two_rois,
    _trim_streamline_in_mask,
    _trim_streamline_endpoints_in_mask,
    _trim_streamline_in_mask_keep_longest,
    compute_streamline_segment,
    cut_streamlines_between_labels,
    cut_streamlines_with_mask,
    get_endpoints_density_map,
    get_head_tail_density_maps,
    CuttingStyle)
from scilpy.image.labels import get_labels_from_mask
from scilpy.tractograms.uncompress import streamlines_to_voxel_coordinates


fetch_data(get_testing_files_dict(), keys=['tractograms.zip'])


def _setup_files():
    """ Load streamlines and masks relevant to the tests here.
    """

    in_ref = os.path.join(SCILPY_HOME, 'tractograms',
                          'streamline_and_mask_operations',
                          'bundle_4_wm.nii.gz')

    in_head_tail = os.path.join(SCILPY_HOME, 'tractograms',
                                'streamline_and_mask_operations',
                                'bundle_4_head_tail.nii.gz')
    in_head_tail_offset = os.path.join(SCILPY_HOME, 'tractograms',
                                       'streamline_and_mask_operations',
                                       'bundle_4_head_tail_offset.nii.gz')
    in_center = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_center.nii.gz')
    in_sft = os.path.join(SCILPY_HOME, 'tractograms',
                          'streamline_and_mask_operations',
                          'bundle_4.tck')

    # Load reference and roi files
    reference = nib.load(in_ref)
    head_tail_rois = nib.load(in_head_tail).get_fdata()
    head_tail_offset_rois = nib.load(in_head_tail_offset).get_fdata()
    center_roi = nib.load(in_center).get_fdata()

    # Load sft
    sft = load_tractogram(in_sft, reference)
    return sft, reference, head_tail_rois, head_tail_offset_rois, center_roi


def test_get_endpoints_density_map():
    """ Test the get_endpoints_density_map function.
    """

    sft, reference, *_ = _setup_files()

    endpoints_map = get_endpoints_density_map(
        sft, point_to_select=1)

    in_result = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_endpoints_1point.nii.gz')

    result = nib.load(in_result).get_fdata()

    assert np.allclose(endpoints_map, result)


def test_get_endpoints_density_map_five_points():
    """ Test the get_endpoints_density_map function with 5 points.
    """

    sft, reference, *_ = _setup_files()

    endpoints_map = get_endpoints_density_map(
        sft, point_to_select=5, to_millimeters=True)

    in_result = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_endpoints_5points.nii.gz')

    result = nib.load(in_result).get_fdata()

    assert np.allclose(endpoints_map, result)


def test_get_head_tail_density_maps():
    """ Test the get_head_tail_density_maps function. This is the same as
    get_endpoints_density_map, but with two outputs. Because the actual tested
    function only adds the two rois together, we do the same here.
    """

    sft, reference, *_ = _setup_files()

    head_map, tail_map = get_head_tail_density_maps(
        sft, point_to_select=1)

    in_result = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_endpoints_1point.nii.gz')

    result = nib.load(in_result).get_fdata()

    assert np.allclose(head_map + tail_map, result)


def test_cut_outside_of_mask_streamlines():
    """ Test the cut_streamlines_with_mask function. This test loads a bundle
    with 10 streamlines, and "cuts it" with a mask that should only keep a
    segment of the bundle.
    """

    sft, reference, _, _, center_roi = _setup_files()

    cut_sft = cut_streamlines_with_mask(sft, center_roi)
    cut_sft.to_vox()
    cut_sft.to_corner()

    in_result = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_cut_center.tck')

    res = load_tractogram(in_result, reference)
    res.to_vox()
    res.to_corner()
    assert np.allclose(cut_sft.streamlines._data, res.streamlines._data)


def test_trim_streamline_in_mask():
    """ Test the _trim_streamline_in_mask function. This function is used by
    cut_streamlines_with_mask, and is tested here to ensure that it works
    correctly. Also provides with code coverage.
    """

    sft, reference, _, _, center_roi = _setup_files()
    sft.to_vox()
    sft.to_corner()

    indices, points_to_idx = streamlines_to_voxel_coordinates(
        sft.streamlines,
        return_mapping=True
    )
    strl_indices = indices[0]
    points_to_indices = points_to_idx[0]

    cut = _trim_streamline_in_mask(
        strl_indices, sft.streamlines[0], points_to_indices, center_roi)

    in_result = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_cut_center.tck')

    res = load_tractogram(in_result, reference)
    res.to_vox()
    res.to_corner()
    assert np.allclose(cut, res.streamlines[0])


def test_cut_outside_of_mask_streamlines_keep_longest():
    """ Test the cut_streamlines_with_mask function with the KEEP_LONGEST
    cutting style. This test loads a bundle with 10 streamlines, and "cuts it"
    with a mask that should only keep the longest segment of the bundle.

    Because we don't have the necessary testing data, this function should
    return the same result as test_cut_outside_of_mask_streamlines.
    """

    sft, reference, _, _, center_roi = _setup_files()

    cut_sft = cut_streamlines_with_mask(
        sft, center_roi, CuttingStyle.KEEP_LONGEST)
    cut_sft.to_vox()
    cut_sft.to_corner()

    in_result = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_cut_center.tck')

    res = load_tractogram(in_result, reference)
    res.to_vox()
    res.to_corner()
    assert np.allclose(cut_sft.streamlines._data, res.streamlines._data)


def test_trim_streamline_in_mask_keep_longest():
    """ Test the _trim_streamline_in_mask_keep_longest function. This function
    is used by cut_streamlines_with_mask, and is tested here to ensure that it
    works correctly. Also provides with code coverage.
    """

    sft, reference, _, _, center_roi = _setup_files()
    sft.to_vox()
    sft.to_corner()

    indices, points_to_idx = streamlines_to_voxel_coordinates(
        sft.streamlines,
        return_mapping=True
    )
    strl_indices = indices[0]
    points_to_indices = points_to_idx[0]

    cut = _trim_streamline_in_mask_keep_longest(
        strl_indices, sft.streamlines[0], points_to_indices, center_roi)

    in_result = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_cut_center.tck')

    res = load_tractogram(in_result, reference)
    res.to_vox()
    res.to_corner()
    assert np.allclose(cut, res.streamlines[0])


def test_cut_outside_of_mask_streamlines_trim_endpoints():
    """ Test the cut_streamlines_with_mask function with the TRIM_ENDPOINTS
    cutting style. This test loads a bundle with 10 streamlines, and "cuts it"
    with a mask that should shave off the endpoints of the bundle.
    """

    sft, reference, _, head_tail_offset_rois, _ = _setup_files()

    cut_sft = cut_streamlines_with_mask(
        sft, head_tail_offset_rois, CuttingStyle.TRIM_ENDPOINTS)
    cut_sft.to_vox()
    cut_sft.to_corner()

    in_result = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_cut_endpoints.tck')

    res = load_tractogram(in_result, reference)
    res.to_vox()
    res.to_corner()
    assert np.allclose(cut_sft.streamlines._data, res.streamlines._data)


def test_trim_streamline_endpoints_in_mask():
    """ Test the _trim_streamline_endpoints_in_mask function. This function is
    used by cut_streamlines_with_mask, and is tested here to ensure that it
    works correctly. Also provides with code coverage.
    """

    sft, reference, _, head_tail_offset_rois, _ = _setup_files()
    sft.to_vox()
    sft.to_corner()

    indices, points_to_idx = streamlines_to_voxel_coordinates(
        sft.streamlines,
        return_mapping=True
    )
    strl_indices = indices[0]
    points_to_indices = points_to_idx[0]

    cut = _trim_streamline_endpoints_in_mask(
        strl_indices, sft.streamlines[0], points_to_indices,
        head_tail_offset_rois)

    in_result = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_cut_endpoints.tck')

    res = load_tractogram(in_result, reference)
    res.to_vox()
    res.to_corner()
    assert np.allclose(cut, res.streamlines[0])


def test_cut_between_labels_streamlines():
    """ Test the cut_between_labels_streamlines function. This test
    loads a bundle with 10 streamlines, and "cuts it" with a mask that
    should not change the bundle.
    """

    sft, reference, head_tail_rois, *_ = _setup_files()
    # head_tail_rois is a mask with two rois that correspond to the bundle's
    # endpoints.
    head_tail_labels = get_labels_from_mask(head_tail_rois)

    cut_sft = cut_streamlines_between_labels(sft, head_tail_labels)
    cut_sft.to_vox()
    cut_sft.to_corner()

    # The expected result is the input bundle.
    in_result = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4.tck')

    res = load_tractogram(in_result, reference)
    res.to_vox()
    res.to_corner()
    # The streamlines should not have changed.
    assert np.allclose(cut_sft.streamlines._data, res.streamlines._data)


def test_cut_between_labels_streamlines_offset():
    """ Test the cut_between_labels_streamlines function. This test
    loads a bundle with 10 streamlines, and cuts it with a mask that
    shaves off the endpoints slightly.
    """

    sft, reference, _, head_tail_offset_rois, _ = _setup_files()
    # head_tail_offset_rois is a mask with two rois that are not exactly at the
    # endpoints of the bundle.
    head_tail_labels = get_labels_from_mask(head_tail_offset_rois)

    cut_sft = cut_streamlines_between_labels(sft, head_tail_labels)

    cut_sft.to_vox()
    cut_sft.to_corner()

    in_result = os.path.join(SCILPY_HOME, 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_cut_endpoints.tck')

    res = load_tractogram(in_result, reference)
    res.to_vox()
    res.to_corner()
    assert np.allclose(cut_sft.streamlines._data, res.streamlines._data)


def test_compute_streamline_segment():
    """ Test the compute_streamline_segment function by cutting a
    streamline between two rois.
    """

    sft, reference, _, head_tail_offset_rois, _ = _setup_files()

    sft.to_vox()
    sft.to_corner()
    one_sft = sft[0]

    # Split head and tail from mask
    roi_data_1, roi_data_2 = split_mask_blobs_kmeans(
        head_tail_offset_rois,
        nb_clusters=2
    )

    (indices, points_to_idx) = streamlines_to_voxel_coordinates(
        one_sft.streamlines,
        return_mapping=True
    )

    strl_indices = indices[0]
    # Find the first and last "voxels" of the streamline that are in the
    # ROIs
    in_strl_idx, out_strl_idx = _intersects_two_rois(roi_data_1,
                                                     roi_data_2,
                                                     strl_indices)
    # If the streamline intersects both ROIs
    if in_strl_idx is not None and out_strl_idx is not None:
        points_to_indices = points_to_idx[0]
        # Compute the new streamline by keeping only the segment between
        # the two ROIs
        res = compute_streamline_segment(one_sft.streamlines[0],
                                         strl_indices,
                                         in_strl_idx, out_strl_idx,
                                         points_to_indices)

    # Streamline should be shorter than the original
    assert len(res) < len(one_sft.streamlines[0])
    assert len(res) == 105
