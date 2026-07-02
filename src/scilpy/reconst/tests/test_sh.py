# -*- coding: utf-8 -*-
import numpy as np
from scilpy.reconst.sh import generate_apodized_delta_kernel


def test_verify_data_vs_sh_order():
    # Quite simple, nothing to test
    pass


def test_compute_sh_coefficients():
    # toDO
    pass


def test_compute_rish():
    # toDO
    pass


def test_peaks_from_sh():
    # toDO
    pass


def test_maps_from_sh():
    # toDO
    pass


def test_convert_sh_basis():
    # toDO
    pass


def test_convert_sh_to_sf():
    # toDO
    pass


def test_generate_apodized_delta_kernel():
    expected_result = np.array([
        1., 0.6350477, 0.6350477, 0.6350477, 0.6350477, 0.6350477,
        0.1838922, 0.1838922, 0.1838922, 0.1838922, 0.1838922, 0.1838922,
        0.1838922, 0.1838922, 0.1838922], dtype=np.float32)
    kernel = generate_apodized_delta_kernel(4, 'descoteaux07', True)
    assert np.allclose(kernel, expected_result)
