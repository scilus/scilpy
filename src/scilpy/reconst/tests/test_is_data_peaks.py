# -*- coding: utf-8 -*-
import os
import nibabel as nib
import numpy as np
from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.reconst.utils import is_data_peaks


def test_is_data_peaks_with_real_data():
    fetch_data(get_testing_files_dict(), keys=['processing.zip'])

    processing_dir = os.path.join(SCILPY_HOME, 'processing')

    # 1. Test with SH data (fODF)
    sh_path = os.path.join(processing_dir, 'fodf_descoteaux07.nii.gz')
    sh_data = nib.load(sh_path).get_fdata()
    assert is_data_peaks(sh_data) is False, "Should identify SH data as False"

    # 2. Test with Peaks data
    peaks_path = os.path.join(processing_dir, 'peaks.nii.gz')
    peaks_data = nib.load(peaks_path).get_fdata()
    assert is_data_peaks(
        peaks_data) is True, "Should identify Peaks data as True"


def test_is_data_peaks_with_edge_cases():
    # 3D data (e.g. 1 directions)
    peaks_3d = np.random.rand(10, 10, 10, 3)
    assert is_data_peaks(peaks_3d) is True

    # SH data with order 4 (15 coefficients) but all zeros
    sh_zeros = np.zeros((10, 10, 10, 15))
    assert is_data_peaks(sh_zeros) is False

    # Data that is clearly peaks (multiple of 3, many zeros)
    peaks_many_zeros = np.zeros((10, 10, 10, 9))
    peaks_many_zeros[5, 5, 5, :3] = [1, 0, 0]
    assert is_data_peaks(peaks_many_zeros) is True
