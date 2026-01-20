# -*- coding: utf-8 -*-
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
import nibabel as nib
import numpy as np
from scilpy.segment.streamlines import streamlines_in_mask, \
    filter_grid_roi_both_ends, filter_grid_roi, filter_ellipsoid, filter_cuboid

# Preparing SFT for all tests here.
# 3 ways to crete ROI, all ~voxels 1 and 2 in x, y, z
# - mask
# - bdo_ellipsoid
# - bdo_cuboid

# Binary mask
mask = np.zeros((10, 10, 10))
mask[1:3, 1:3, 1:3] = 1

# bdo.
# Note. Everything else here is in vox, corner, except bdo_center
bdo_center_rasmm_centerorigin = np.asarray([1.5, 1.5, 1.5])
bdo_radius_mm = 1.5

# Preparing streamlines fitting criteria.
# line 0 'all': entirely in mask.
line0 = [[1.5, 1.5, 1.5],
         [1.6, 1.6, 1.6],
         [1.7, 1.7, 1.7],
         [2.5, 2.5, 2.5]]

# line 1 'both_ends': both ends in mask but not in the middle.
line1 = [[1.5, 1.5, 1.5],
         [9, 9, 9],
         [1.7, 1.7, 1.7],
         [2.5, 2.5, 2.5]]

# line 2 'any': Only one end in mask
line2 = [[1.5, 1.5, 1.5],
         [1.6, 1.6, 1.6],
         [1.7, 1.7, 1.7],
         [9, 9, 9]]

# line 3 'any': Both ends out of mask but touches in the middle
line3 = [[9, 9, 9],
         [1.6, 1.6, 1.6],
         [1.7, 1.7, 1.7],
         [9, 9, 9]]

# line 4 (exclude): never in mask
line4 = [[9, 9, 9.0],
         [8, 8, 8.0],
         [7, 7, 7.0],
         [9, 9, 9.0]]

# line 5 'any': passes through, but no real point inside.
line5 = [[0.1, 0.1, 0.1],
         [8, 8, 8.0],
         [7, 7, 7.0],
         [9, 9, 9.0]]

fake_reference = nib.Nifti1Image(
    np.zeros((10, 10, 10, 1)), affine=np.eye(4))
sft = StatefulTractogram([line0, line1, line2, line3, line4, line5],
                         fake_reference, space=Space.VOXMM,
                         origin=Origin('corner'))


def test_streamlines_in_mask():
    # Test option 'all'
    ids = streamlines_in_mask(sft, mask, all_in=True)
    assert np.array_equal(ids, [0])

    # Test option 'any'
    ids = streamlines_in_mask(sft, mask, all_in=False)
    assert np.array_equal(ids, [0, 1, 2, 3, 5])


def test_filter_grid_roi_both_ends():
    # Pretending to have two masks
    # Test option 'both ends'
    new_sft, ids = filter_grid_roi_both_ends(sft, mask_1=mask, mask_2=mask)
    assert np.array_equal(ids, [0, 1])
    assert len(new_sft) == 2


def test_filter_grid_roi():
    # Note. Distance not tested yet. (toDo)
    roi_options = (mask,)
    _test_all_criteria(filter_grid_roi, roi_options)


def test_filter_ellipsoid():
    roi_options = (bdo_radius_mm, bdo_center_rasmm_centerorigin)
    _test_all_criteria(filter_ellipsoid, roi_options)


def test_filter_cuboid():
    roi_options = (bdo_radius_mm, bdo_center_rasmm_centerorigin)
    _test_all_criteria(filter_cuboid, roi_options)


def _test_all_criteria(fct, roi_options):
    """
    The three filtering methods (filter_grid_roi, filter_ellipsoid,
    filter_cuboid) test the same criteria, but with a different way to treat
    the ROI.
    """
    # Parameter is "is_exclude", so:
    include=False
    exclude=True

    # Test 'any'
    # Testing the returned sft only this once
    ids, new_sft = fct(sft, *roi_options, 'any', include)
    assert np.array_equal(ids, [0, 1, 2, 3, 5])
    assert len(new_sft) == 5
    ids, _ =  fct(sft, *roi_options, 'any', exclude)
    assert np.array_equal(ids, [4])

    # Test 'all'
    ids, _ =  fct(sft, *roi_options, 'all', include)
    assert np.array_equal(ids, [0])
    ids, _ =  fct(sft, *roi_options, 'all', exclude)
    assert np.array_equal(ids, [1, 2, 3, 4, 5])

    # Test 'either_end'
    ids, _ =  fct(sft, *roi_options, 'either_end', include)
    assert np.array_equal(ids, [0, 1, 2])
    ids, _ =  fct(sft, *roi_options, 'either_end', exclude)
    assert np.array_equal(ids, [3, 4, 5])

    # Test 'both_ends'
    ids, _ =  fct(sft, *roi_options, 'both_ends', include)
    assert np.array_equal(ids, [0, 1])
    ids, _ =  fct(sft, *roi_options, 'both_ends', exclude)
    assert np.array_equal(ids, [2, 3, 4, 5])