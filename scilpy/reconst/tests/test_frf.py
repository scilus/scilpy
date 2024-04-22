# -*- coding: utf-8 -*-
import os
import tempfile

import nibabel as nib
import numpy as np
from dipy.io import read_bvals_bvecs

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.reconst.frf import compute_ssst_frf

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
    # toDO
    pass


def test_replace_frf():
    # toDo
    pass
