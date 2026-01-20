# -*- coding: utf-8 -*-
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
import nibabel as nib
import numpy as np
from scilpy.segment.streamlines import streamlines_in_mask, \
    filter_grid_roi_both_ends, filter_grid_roi

# Preparing SFT for all tests here.
# Binary mask: ROI on x=1 and x=2
mask = np.zeros((10, 10, 10))
mask[1:3, :, :] = 1

# line 0 entirely in mask. y and z values are random.
line0 = [[1.3, 5, 6],
         [1.5, 5, 7],
         [2.5, 7, 4],
         [2.9, 9, 9]]

# line 1: partially in mask (both ends in mask but not in the middle)
line1 = [[1.3, 5, 6],
         [5.5, 5, 7],
         [2.5, 7, 4],
         [2.9, 9, 9]]

# line 2: Only one end in mask
line2 = [[1.3, 5, 6],
         [5.5, 5, 7],
         [6.5, 7, 4],
         [7.9, 9, 9]]

# line 3: Both ends out of mask but touches in the middle
line3 = [[5.3, 5, 6],
         [1.5, 5, 7],
         [2.5, 7, 4],
         [7.9, 9, 9]]

# line 4: never in mask
line4 = [[4.3, 5, 6],
         [5.5, 5, 7],
         [7.5, 7, 4],
         [9.9, 9, 9]]

# line 5: passes through, but no real point inside.
line5 = [[0.3, 5, 6],
         [5.5, 5, 7],
         [7.5, 7, 4],
         [9.9, 9, 9]]

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
    # Testing the returned SFT only the first time.
    # Parameter is "is_exclude", so:
    include=False
    exclude=True

    # Test 'any'
    ids, new_sft, rejected_sft = filter_grid_roi(
        sft, mask, 'any', include,
        return_sft=True, return_rejected_sft=True)
    assert np.array_equal(ids, [0, 1, 2, 3, 5])
    assert len(new_sft) == 5
    ids = filter_grid_roi(sft, mask, 'any', exclude)
    assert np.array_equal(ids, [4])

    # Test 'all'
    ids = filter_grid_roi(sft, mask, 'all', include)
    assert np.array_equal(ids, [0])
    ids = filter_grid_roi(sft, mask, 'all', exclude)
    assert np.array_equal(ids, [1, 2, 3, 4, 5])

    # Test 'either_end'
    ids = filter_grid_roi(sft, mask, 'either_end', include)
    assert np.array_equal(ids, [0, 1, 2])
    ids = filter_grid_roi(sft, mask, 'either_end', exclude)
    assert np.array_equal(ids, [3, 4, 5])

    # Test 'both_ends'
    ids = filter_grid_roi(sft, mask, 'both_ends', include)
    assert np.array_equal(ids, [0, 1])
    ids = filter_grid_roi(sft, mask, 'both_ends', exclude)
    assert np.array_equal(ids, [2, 3, 4, 5])