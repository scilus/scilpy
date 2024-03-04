# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np

from scilpy.dwi.utils import extract_dwi_shell, extract_b0
from scilpy.gradients.bvec_bval_tools import B0ExtractionStrategy


def test_extract_dwi_shell():
    # DWI with 5 gradients. Values for gradient #i are all i.
    dwi = np.ones((10, 10, 10, 5))
    bvecs = np.ones((5, 3))
    for i in range(5):
        dwi[..., i] = i
        bvecs[i, :] = i
    bvals = np.asarray([0, 1010, 12, 990, 2000])

    # Note. Not testing the block_size option.
    dwi_img = nib.Nifti1Image(dwi, affine=np.eye(4))
    indices, shell_data, output_bvals, output_bvecs = extract_dwi_shell(
        dwi_img, bvals, bvecs, bvals_to_extract=[0, 2000], tol=15,
        block_size=None)
    assert np.array_equal(indices, [0, 2, 4])
    assert np.array_equal(shell_data[0, 0, 0, :], [0, 2, 4])
    assert np.array_equal(output_bvals, [0, 12, 2000])
    assert np.array_equal(output_bvecs[:, 0], [0, 2, 4])


def test_extract_b0():
    # DWI with 5 gradients. Values for gradient #i are all i.
    dwi = np.ones((10, 10, 10, 5))
    for i in range(5):
        dwi[..., i] = i
    b0_mask = np.asarray([0, 1, 0, 1, 1], dtype=bool)
    dwi_img = nib.Nifti1Image(dwi, affine=np.eye(4))

    # Note. Not testing the block_size option.

    # Test 1: Take the first.
    strategy = B0ExtractionStrategy.FIRST
    b0_data = extract_b0(dwi_img, b0_mask, strategy=strategy,
                         extract_in_cluster=False, block_size=None)
    assert len(b0_data.shape) == 3  # Should be 3D; one b-value
    assert b0_data[0, 0, 0] == 1

    # Test 2: Take the first, per continuous cluser
    b0_data = extract_b0(dwi_img, b0_mask, strategy=strategy,
                         extract_in_cluster=True, block_size=None)
    assert b0_data.shape[-1] == 2
    assert np.array_equal(b0_data[0, 0, 0, :], [1, 3])

    # Test 3: Take the mean
    strategy = B0ExtractionStrategy.MEAN
    b0_data = extract_b0(dwi_img, b0_mask, strategy=strategy,
                         extract_in_cluster=False, block_size=None)
    assert len(b0_data.shape) == 3  # Should be 3D; one b-value
    assert b0_data[0, 0, 0] == np.mean([1, 3, 4])

    # Test 4: Take the mean per cluster
    b0_data = extract_b0(dwi_img, b0_mask, strategy=strategy,
                         extract_in_cluster=True, block_size=None)
    assert b0_data.shape[-1] == 2
    assert b0_data[0, 0, 0, 0] == 1
    assert b0_data[0, 0, 0, 1] == np.mean([3, 4])

    # Test 5: Take all
    strategy = B0ExtractionStrategy.ALL
    b0_data = extract_b0(dwi_img, b0_mask, strategy=strategy,
                         extract_in_cluster=False, block_size=None)
    assert b0_data.shape[-1] == 3
    assert np.array_equal(b0_data[0, 0, 0, :], [1, 3, 4])
