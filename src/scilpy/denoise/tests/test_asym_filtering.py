import numpy as np

from scilpy.denoise.asym_filtering import \
        unified_filtering, cosine_filtering
from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.tests.arrays import (
    fodf_3x3_order8_descoteaux07,
    fodf_3x3_order8_descoteaux07_filtered_unified,
    fodf_3x3_order8_descoteaux07_filtered_cosine)


def test_unified_asymmetric_filtering():
    """
    Test AsymmetricFilter on a simple 3x3 grid.
    """
    in_sh = fodf_3x3_order8_descoteaux07
    sh_basis = 'descoteaux07'
    legacy = True
    sphere_str = 'repulsion100'
    sigma_spatial = 1.0
    sigma_align = 0.8
    sigma_range = 0.2
    sigma_angle = 0.06
    win_hwidth = 3
    exclude_center = False
    device_type = 'cpu'
    use_opencl = False

    sh_order, full_basis = get_sh_order_and_fullness(in_sh.shape[-1])
    asym_sh = unified_filtering(in_sh, sh_order, sh_basis,
                                is_legacy=legacy,
                                full_basis=full_basis,
                                sphere_str=sphere_str,
                                sigma_spatial=sigma_spatial,
                                sigma_align=sigma_align,
                                sigma_angle=sigma_angle,
                                rel_sigma_range=sigma_range,
                                win_hwidth=win_hwidth,
                                exclude_center=exclude_center,
                                device_type=device_type,
                                use_opencl=use_opencl)

    assert np.allclose(asym_sh, fodf_3x3_order8_descoteaux07_filtered_unified)


def test_cosine_filtering():
    """
    Test cosine filtering on a simple 3x3 grid.
    """
    in_sh = fodf_3x3_order8_descoteaux07
    sh_basis = 'descoteaux07'
    legacy = True
    sphere_str = 'repulsion100'
    sigma_spatial = 1.0
    sharpness = 1.0

    sh_order, full_basis = get_sh_order_and_fullness(in_sh.shape[-1])
    out = cosine_filtering(in_sh, sh_order, sh_basis, full_basis, legacy,
                           sharpness, sphere_str, sigma_spatial)

    assert np.allclose(out, fodf_3x3_order8_descoteaux07_filtered_cosine)
