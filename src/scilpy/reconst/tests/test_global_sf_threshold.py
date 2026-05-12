import os
import numpy as np
import nibabel as nib
import pytest
from dipy.data import get_sphere
from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.reconst.utils import compute_sf_threshold_mask

def test_compute_sf_threshold_mask_real_data():
    # Fetch data
    fetch_data(get_testing_files_dict(), keys=['processing.zip'])
    sh_path = os.path.join(SCILPY_HOME, 'processing', 'sh_1000.nii.gz')

    # Load data
    img = nib.load(sh_path)
    data = img.get_fdata(dtype=np.float32)
    sphere = get_sphere(name='repulsion724')

    # 1. Relative threshold tests
    mask0, _, _ = compute_sf_threshold_mask(data, sphere, relative_factor=0.0)
    count0 = np.sum(mask0)

    mask01, _, _ = compute_sf_threshold_mask(data, sphere, relative_factor=0.1)
    count01 = np.sum(mask01)

    mask1, _, _ = compute_sf_threshold_mask(data, sphere, relative_factor=1.0)
    count1 = np.sum(mask1)

    assert count0 >= count01 >= count1, "Relative threshold counts not monotonic"

    # 2. Absolute threshold tests
    mask_abs_low, _, _ = compute_sf_threshold_mask(data, sphere, absolute_threshold=0.01)
    count_abs_low = np.sum(mask_abs_low)

    mask_abs_high, _, _ = compute_sf_threshold_mask(data, sphere, absolute_threshold=0.1)
    count_abs_high = np.sum(mask_abs_high)

    assert count_abs_low >= count_abs_high, "Absolute threshold counts not monotonic"
