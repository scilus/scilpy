import pytest

from scilpy.tests.arrays import fodf_3x3_order8_descoteaux07, \
    fodf_3x3_order8_descoteaux07_filtered_unified


@pytest.mark.ml
def test_get_data_order_6_28coeffs():
    """ fODF should keep the same number of coeffs."""

    # Imported here so as not to trigger an import error with pytorch
    from scilpy.ml.bundleparc.utils import get_data
    # Not actually order 8
    fodf_order6 = fodf_3x3_order8_descoteaux07
    data = get_data(fodf_order6, 28)

    assert data.shape[0] == 28


@pytest.mark.ml
def test_get_data_order_6_45coeffs():
    """ fODF should be padded with zeros."""

    # Imported here so as not to trigger an import error with pytorch
    from scilpy.ml.bundleparc.utils import get_data
    # Not actually order 8
    fodf_order6 = fodf_3x3_order8_descoteaux07
    data = get_data(fodf_order6, 45)

    assert data.shape[0] == 45


@pytest.mark.ml
def test_get_data_order_8_28coeffs():
    """ fODF should be truncated """

    # Imported here so as not to trigger an import error with pytorch
    from scilpy.ml.bundleparc.utils import get_data
    # Not actually order 8
    fodf_order8 = fodf_3x3_order8_descoteaux07_filtered_unified[..., :45]
    data = get_data(fodf_order8, 26)

    assert data.shape[0] == 26
