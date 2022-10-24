import numpy as np

from scilpy.denoise.bilateral_filtering import \
        angle_aware_bilateral_filtering_cpu
from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.tests.arrays import (
    fodf_3x3_order8_descoteaux07, fodf_3x3_order8_descoteaux07_filtered)


def test_angle_aware_bilateral_filtering_cpu_one_process():
    """ Test angle_aware_bilateral_filtering_cpu on a simple 3x3 grid
    using one process.
    """

    in_sh = fodf_3x3_order8_descoteaux07
    sh_order = 8
    sh_basis = 'descoteaux07'
    in_full_basis = False
    sphere_str = 'repulsion100'
    sigma_spatial = 1.0
    sigma_angular = 1.0
    sigma_range = 1.0
    nbr_processes = 1.0

    sh_order, full_basis = get_sh_order_and_fullness(in_sh.shape[-1])
    out = angle_aware_bilateral_filtering_cpu(in_sh, sh_order,
                                              sh_basis, in_full_basis,
                                              sphere_str, sigma_spatial,
                                              sigma_angular, sigma_range,
                                              nbr_processes)

    assert np.allclose(out, fodf_3x3_order8_descoteaux07_filtered)


def test_angle_aware_bilateral_filtering_cpu_four_processes():
    """ Test angle_aware_bilateral_filtering_cpu on a simple 3x3 grid
    using four processes.
    """

    in_sh = fodf_3x3_order8_descoteaux07
    sh_order = 8
    sh_basis = 'descoteaux07'
    in_full_basis = False
    sphere_str = 'repulsion100'
    sigma_spatial = 1.0
    sigma_angular = 1.0
    sigma_range = 1.0
    nbr_processes = 4

    sh_order, full_basis = get_sh_order_and_fullness(in_sh.shape[-1])
    out = angle_aware_bilateral_filtering_cpu(in_sh, sh_order,
                                              sh_basis, in_full_basis,
                                              sphere_str, sigma_spatial,
                                              sigma_angular, sigma_range,
                                              nbr_processes)

    assert np.allclose(out, fodf_3x3_order8_descoteaux07_filtered)
