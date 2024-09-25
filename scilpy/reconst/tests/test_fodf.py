# -*- coding: utf-8 -*-
import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix

from scilpy.reconst.fodf import get_ventricles_max_fodf
from scilpy.reconst.utils import find_order_from_nb_coeff
from scilpy.tests.arrays import fodf_3x3_order8_descoteaux07


def test_get_ventricles_max_fodf():
    fake_fa = np.ones((3, 3, 1))  # High FA
    fake_fa[1:3, 0:2, 0] = 0      # Low in ventricles
    fake_md = np.zeros((3, 3, 1))  # Low MD
    fake_md[0:2, 0:2, 0] = 1       # High in ventricles
    zoom = [1, 1, 1]
    fa_threshold = 0.5
    md_threshold = 0.5
    sh_basis = 'descoteaux07'

    # Should find that the only 2 ventricle voxels are at [1, 0:2, 0]
    mean, mask = get_ventricles_max_fodf(
        fodf_3x3_order8_descoteaux07, fake_fa, fake_md, zoom, sh_basis,
        fa_threshold, md_threshold, small_dims=True)

    expected_mask = np.logical_and(~fake_fa.astype(bool), fake_md)
    assert np.count_nonzero(mask) == 2
    assert np.array_equal(mask.astype(bool), expected_mask)

    # Reconstruct SF values same as in method.
    order = find_order_from_nb_coeff(fodf_3x3_order8_descoteaux07)
    sphere = get_sphere('repulsion100')
    b_matrix, _ = sh_to_sf_matrix(sphere, order, sh_basis, legacy=True)

    sf1 = np.dot(fodf_3x3_order8_descoteaux07[1, 0, 0], b_matrix)
    sf2 = np.dot(fodf_3x3_order8_descoteaux07[1, 1, 0], b_matrix)

    assert mean == np.mean([np.max(sf1), np.max(sf2)])


def test_fit_from_model():
    # toDO
    pass


def test_verify_failed_voxels_shm_coeff():
    # Quite simple, nothing to test
    pass


def test_verify_frf_files():
    # Quite simple, nothing to test
    pass
