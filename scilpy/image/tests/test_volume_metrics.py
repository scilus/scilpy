# -*- coding: utf-8 -*-
import os

import nibabel as nib
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.image.volume_metrics import estimate_piesno_sigma

fetch_data(get_testing_files_dict(), keys=['processing.zip'])

def test_estimate_piesno_sigma():
    #  Piesno itself is from dipy. Not testing.
    #  Testing that the result is always the same as today's

    data_path =  os.path.join(SCILPY_HOME, 'processing',
                              'dwi_crop_1000.nii.gz')
    data = nib.load(data_path).get_fdata(dtype=np.float32)
    piesno, mask_noise = estimate_piesno_sigma(data, number_coils=1)

    assert len(piesno) == data.shape[2]   # One sigma per SLICE
    assert np.array_equal(mask_noise.shape, data.shape[0:3])
    assert np.count_nonzero(mask_noise) == 656
    assert np.allclose(piesno[0], 1.2512785)
