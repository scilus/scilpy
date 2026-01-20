# -*- coding: utf-8 -*-
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
import nibabel as nib
import numpy as np
from scilpy.segment.streamlines import streamlines_in_mask


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
