# -*- coding: utf-8 -*-
import os
import tempfile

import nibabel as nib
import numpy as np
from dipy.io import read_bvals_bvecs

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.reconst.frf import compute_ssst_frf, compute_msmt_frf, replace_frf

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()
in_dwi = os.path.join(SCILPY_HOME, 'processing', 'dwi_crop.nii.gz')
in_bval = os.path.join(SCILPY_HOME, 'processing', 'dwi.bval')
in_bvec = os.path.join(SCILPY_HOME, 'processing', 'dwi.bvec')


def test_compute_ssst_frf():
    # Uses data from our test data.
    # To use a smaller subset, we need to ensure that it has at least one
    # voxel with FA higher than 0.7. Quite fast as is, so, ok.
    dwi = nib.load(in_dwi).get_fdata()  # Shape: 57, 67, 56, 64
    bvals, bvecs = read_bvals_bvecs(in_bval, in_bvec)

    result = compute_ssst_frf(dwi, bvals, bvecs)

    # Value with current data at the date of test creation:
    expected_result = [1.03068237e-03, 2.44994949e-04,
                       2.44994949e-04, 3.26903486e+03]
    assert np.allclose(result, expected_result)


def test_compute_msmt_frf():
    # Uses data from our test data.
    # To use a smaller subset, we need to ensure that it has at least one
    # voxel with each tissue type.
    dwi = nib.load(in_dwi).get_fdata()  # Shape: 57, 67, 56, 64
    bvals, bvecs = read_bvals_bvecs(in_bval, in_bvec)

    responses, masks = compute_msmt_frf(dwi, bvals, bvecs)

    # Value with current data at the date of test creation:
    expected_result_wm = [[1.56925332e-03, 4.68706503e-04,
                           4.68706503e-04, 3.26903486e+03],
                          [1.15181122e-03, 3.75303294e-04,
                           3.75303294e-04, 3.26903486e+03],
                          [8.61299793e-04, 3.14541494e-04,
                           3.14541494e-04, 3.26903486e+03]]
    expected_result_gm = [[9.74471606e-04, 8.34628732e-04,
                           8.34628732e-04, 3.42007686e+03],
                          [7.76991313e-04, 6.89550835e-04,
                           6.89550835e-04, 3.42007686e+03],
                          [6.26617550e-04, 5.73389066e-04,
                           5.73389066e-04, 3.42007686e+03]]
    expected_result_csf = [[9.33140592e-04, 8.31445917e-04,
                            8.31445917e-04, 3.62805637e+03],
                           [7.69894406e-04, 7.07255607e-04,
                            7.07255607e-04, 3.62805637e+03],
                           [6.34735398e-04, 5.96451860e-04,
                            5.96451860e-04, 3.62805637e+03]]
    assert np.allclose(responses[0], expected_result_wm)
    assert np.allclose(responses[1], expected_result_gm)
    assert np.allclose(responses[2], expected_result_csf)

    assert np.count_nonzero(masks[0]) == 845   # wm
    assert np.count_nonzero(masks[1]) == 1779  # gm
    assert np.count_nonzero(masks[2]) == 449   # csf


def test_replace_frf():
    old_frf = np.random.rand(4)
    new_frf = "15,4,4"
    result = replace_frf(old_frf, new_frf, no_factor=True)

    # Rounds to float64
    assert np.allclose(result, [15, 4, 4, old_frf[-1]])
