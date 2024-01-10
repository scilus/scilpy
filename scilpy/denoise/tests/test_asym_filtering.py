import numpy as np

from scilpy.denoise.asym_filtering import \
        angle_aware_bilateral_filtering_cpu, cosine_filtering
from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.tests.arrays import (
    fodf_3x3_order8_descoteaux07,
    fodf_3x3_order8_descoteaux07_filtered_bilateral,
    fodf_3x3_order8_descoteaux07_filtered_cosine)


def test_angle_aware_bilateral_filtering():
    """
    Test angle_aware_bilateral_filtering_cpu on a simple 3x3 grid.
    """
    in_sh = fodf_3x3_order8_descoteaux07
    sh_basis = 'descoteaux07'
    sphere_str = 'repulsion100'
    sigma_spatial = 1.0
    sigma_angular = 1.0
    sigma_range = 1.0

    sh_order, full_basis = get_sh_order_and_fullness(in_sh.shape[-1])
    out = angle_aware_bilateral_filtering_cpu(in_sh, sh_order,
                                              sh_basis, full_basis,
                                              sphere_str, sigma_spatial,
                                              sigma_angular, sigma_range)

    assert np.allclose(out, fodf_3x3_order8_descoteaux07_filtered_bilateral)


def test_cosine_filtering():
    """
    Test cosine filtering on a simple 3x3 grid.
    """
    in_sh = fodf_3x3_order8_descoteaux07
    sh_basis = 'descoteaux07'
    sphere_str = 'repulsion100'
    sigma_spatial = 1.0
    sharpness = 1.0

    sh_order, full_basis = get_sh_order_and_fullness(in_sh.shape[-1])
    out = cosine_filtering(in_sh, sh_order, sh_basis, full_basis,
                           sharpness, sphere_str, sigma_spatial)

    assert np.allclose(out, fodf_3x3_order8_descoteaux07_filtered_cosine)
