# -*- coding: utf-8 -*-
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
import nibabel as nib
import numpy as np
from scilpy.segment.streamlines import streamlines_in_mask, \
    filter_grid_roi_both_ends


def test_streamlines_in_mask():
    # Binary mask: ROI on x=1 and x=2
    mask = np.zeros((10, 10, 10))
    mask[1:3, :, :] = 1
    fake_reference = nib.Nifti1Image(
        np.zeros((10, 10, 10, 1)), affine=np.eye(4))

    # line 1 entirely in mask. y and z values are random.
    line1 = [[1.3, 5, 6],
             [1.5, 5, 7],
             [2.5, 7, 4],
             [2.9, 9, 9]]

    # line 2: partially in mask.
    line2 = [[1.3, 5, 6],
             [5.5, 5, 7],
             [2.5, 7, 4],
             [2.9, 9, 9]]

    # line3: never in mask
    line3 = [[4.3, 5, 6],
             [5.5, 5, 7],
             [7.5, 7, 4],
             [9.9, 9, 9]]

    sft = StatefulTractogram([line1, line2, line3], fake_reference,
                             space=Space.VOXMM, origin=Origin('corner'))

    # Test option 'all'
    ids = streamlines_in_mask(sft, mask, all_in=True)
    assert len(ids) == 1
    assert ids[0] == 0

    # Test option 'any'
    ids = streamlines_in_mask(sft, mask, all_in=False)
    assert len(ids) == 2
    assert ids[0] == 0
    assert ids[1] == 1


def test_filter_grid_roi_both_ends():
    # Binary mask: ROI on x=1 and x=2
    mask = np.zeros((10, 10, 10))
    mask[1:3, :, :] = 1
    fake_reference = nib.Nifti1Image(
        np.zeros((10, 10, 10, 1)), affine=np.eye(4))

    # line 1 entirely in mask. y and z values are random.
    line1 = [[1.3, 5, 6],
             [1.5, 5, 7],
             [2.5, 7, 4],
             [2.9, 9, 9]]

    # line 2: both ends in mask but not in the middle
    line2 = [[1.3, 5, 6],
             [5.5, 5, 7],
             [2.5, 7, 4],
             [2.9, 9, 9]]

    # line 3: Only one end in mask
    line3 = [[1.3, 5, 6],
             [5.5, 5, 7],
             [6.5, 7, 4],
             [7.9, 9, 9]]

    # line 4: Both ends out of mask
    line4 = [[1.3, 5, 6],
             [5.5, 5, 7],
             [6.5, 7, 4],
             [7.9, 9, 9]]

    sft = StatefulTractogram([line1, line2, line3, line4], fake_reference,
                             space=Space.VOXMM, origin=Origin('corner'))

    new_sft, ids = filter_grid_roi_both_ends(sft, mask_1=mask, mask_2=mask)
    assert len(ids) == 2
    assert len(new_sft) == 2
    assert ids[0] == 0
    assert ids[1] == 1
    