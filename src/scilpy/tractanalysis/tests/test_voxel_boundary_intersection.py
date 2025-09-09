# -*- coding: utf-8 -*-
import numpy as np
import nibabel as nib
from dipy.io.stateful_tractogram import StatefulTractogram, Origin, Space

from scilpy.tractanalysis.voxel_boundary_intersection import \
    subdivide_streamlines_at_voxel_faces


def test_subdivide():
    # Creating a few very short streamlines (2 points).

    # streamline1: stays inside its voxel
    streamline1 = np.asarray([[23.0, 23.0, 23.0],
                              [23.5, 23.5, 23.5]], dtype=np.float32)

    # streamline2: changes to next voxel on second dim only. Covers 2 voxels
    streamline2 = np.asarray([[23.0, 23.1, 23.0],
                              [23.5, 22.2, 23.5]], dtype=np.float32)

    # streamline3: covers 11 voxels
    streamline3 = np.asarray([[0.0, 0.0, 0.0],
                              [10.99, 0.0, 0.0]], dtype=np.float32)

    fake_ref = nib.Nifti1Image(np.zeros((3, 3, 3)), affine=np.eye(4))
    sft = StatefulTractogram([streamline1, streamline2,
                              streamline3], reference=fake_ref,
                             origin=Origin('corner'), space=Space('vox'))
    split = subdivide_streamlines_at_voxel_faces(sft.streamlines)

    # Streamline 1 should stay the same
    assert np.array_equal(split[0], streamline1)

    # Streamline 2 should be split into 2 (one more point)
    assert split[1].shape[0] == 3
    assert np.array_equal(split[1][0, :], streamline2[0, :])
    assert np.array_equal(split[1][-1, :], streamline2[-1, :])

    # streamline 3 should be split into 11 (12 points)
    assert split[2].shape[0] == 12
