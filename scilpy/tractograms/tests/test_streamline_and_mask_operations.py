# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np

from dipy.io.streamline import load_tractogram, save_tractogram

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home
from scilpy.tractograms.streamline_and_mask_operations import (
    cut_between_masks_streamlines,
    cut_outside_of_mask_streamlines,
    get_endpoints_density_map,
    get_head_tail_density_maps)

fetch_data(get_testing_files_dict(), keys=['tractograms.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def _setup_files():
    """ Load streamlines and masks relevant to the tests here.
    """
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_ref = os.path.join(get_home(), 'tractograms',
                          'streamline_and_mask_operations',
                          'bundle_4_wm.nii.gz')

    in_head_tail = os.path.join(get_home(), 'tractograms',
                                'streamline_and_mask_operations',
                                'bundle_4_head_tail.nii.gz')
    in_head_tail_offset = os.path.join(get_home(), 'tractograms',
                                       'streamline_and_mask_operations',
                                       'bundle_4_head_tail_offset.nii.gz')
    in_center = os.path.join(get_home(), 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_center.nii.gz')
    in_sft = os.path.join(get_home(), 'tractograms',
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

    sft, reference, _, _, _ = \
        _setup_files()

    sft.to_vox()

    endpoints_map = get_endpoints_density_map(
        sft.streamlines, reference.shape, point_to_select=1)

    os.chdir(os.path.expanduser(tmp_dir.name))
    in_result = os.path.join(get_home(), 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_endpoints_1point.nii.gz')

    result = nib.load(in_result).get_fdata()
    assert np.allclose(endpoints_map, result)


def test_get_endpoints_density_map_five_points():
    """ Test the get_endpoints_density_map function with 5 points.
    """

    sft, reference, _, _, _ = \
        _setup_files()

    sft.to_vox()

    endpoints_map = get_endpoints_density_map(
        sft.streamlines, reference.shape, point_to_select=5)

    os.chdir(os.path.expanduser(tmp_dir.name))
    in_result = os.path.join(get_home(), 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_endpoints_5points.nii.gz')

    result = nib.load(in_result).get_fdata()
    assert np.allclose(endpoints_map, result)


def test_get_head_tail_density_maps():
    """ Test the get_head_tail_density_maps function. This is the same as
    get_endpoints_density_map, but with two outputs. Because the actual tested
    function only adds the two rois together, we do the same here.
    """

    sft, reference, _, _, _ = \
        _setup_files()

    sft.to_vox()

    head_map, tail_map = get_head_tail_density_maps(
        sft.streamlines, reference.shape, point_to_select=1)

    os.chdir(os.path.expanduser(tmp_dir.name))
    in_result = os.path.join(get_home(), 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_endpoints_1point.nii.gz')

    result = nib.load(in_result).get_fdata()
    assert np.allclose(head_map + tail_map, result)


def test_cut_outside_of_mask_streamlines():

    sft, reference, _, _, center_roi = \
        _setup_files()

    cut_sft = cut_outside_of_mask_streamlines(
        sft, center_roi)

    os.chdir(os.path.expanduser(tmp_dir.name))
    in_result = os.path.join(get_home(), 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_cut_center.tck')

    res = load_tractogram(in_result, reference)
    # `cut_outside_of_mask_streamlines` always returns a voxel space sft
    # with streamlines in corner, so move the expected result to the same
    # space.
    res.to_vox()
    res.to_corner()
    assert np.allclose(cut_sft.streamlines._data, res.streamlines._data)


def test_cut_between_masks_streamlines():
    """ Test the cut_between_masks_streamlines function. This test
    loads a bundle with 10 streamlines, and "cuts it" with a mask that
    should not changed the bundle.
    """

    sft, reference, head_tail_rois, _, _ = \
        _setup_files()
    # head_tail_rois is a mask with two rois that correspond
    # to the bundle's endpoints.
    cut_sft = cut_between_masks_streamlines(
        sft, head_tail_rois)

    os.chdir(os.path.expanduser(tmp_dir.name))
    # The expected result is the input bundle.
    in_result = os.path.join(get_home(), 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4.tck')
    save_tractogram(cut_sft, in_result)
    res = load_tractogram(in_result, reference)
    # `cut_between_masks_streamlines` always returns a voxel space sft
    # with streamlines in corner, so move the expected result to the same
    # space.
    res.to_vox()
    res.to_corner()
    # The streamlines should not have changed.
    assert np.allclose(cut_sft.streamlines._data, res.streamlines._data)


def test_cut_between_masks_streamlines_offset():
    """ Test the cut_between_masks_streamlines function. This test
    loads a bundle with 10 streamlines, and cuts it with a mask that
    shaves off the endpoints slightly.
    """

    sft, reference, _, head_tail_offset_rois, _ = \
        _setup_files()
    # head_tail_offset_rois is a mask with two rois that are not
    # exactly at the endpoints of the bundle.
    cut_sft = cut_between_masks_streamlines(
        sft, head_tail_offset_rois)

    os.chdir(os.path.expanduser(tmp_dir.name))
    in_result = os.path.join(get_home(), 'tractograms',
                             'streamline_and_mask_operations',
                             'bundle_4_cut_endpoints.tck')
    save_tractogram(cut_sft, in_result)
    res = load_tractogram(in_result, reference)
    # `cut_between_masks_streamlines` always returns a voxel space sft
    # with streamlines in corner, so move the expected result to the same
    # space.
    res.to_vox()
    res.to_corner()
    assert np.allclose(cut_sft.streamlines._data, res.streamlines._data)
