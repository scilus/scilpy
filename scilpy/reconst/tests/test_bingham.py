# -*- coding: utf-8 -*-
import numpy as np
from dipy.core.sphere import hemi_icosahedron

from scilpy.tests.arrays import (fodf_3x3_order8_descoteaux07,
                                 fodf_3x3_bingham, fodf_3x3_bingham_sf,
                                 fodf_3x3_bingham_fd, fodf_3x3_bingham_fs,
                                 fodf_3x3_bingham_ff, fodf_3x3_bingham_peaks)
from scilpy.reconst.bingham import (bingham_fit_sh, bingham_to_sf,
                                    bingham_to_peak_direction,
                                    compute_fiber_density,
                                    compute_fiber_fraction,
                                    compute_fiber_spread)


def test_bingham_to_sf():
    in_bingham = fodf_3x3_bingham.copy()
    sphere = hemi_icosahedron

    sf = bingham_to_sf(in_bingham, sphere.vertices)
    assert np.allclose(sf, fodf_3x3_bingham_sf)


def test_bingham_to_peak_direction():
    bingham = fodf_3x3_bingham.copy()

    peaks = bingham_to_peak_direction(bingham)
    assert np.allclose(peaks, fodf_3x3_bingham_peaks)


def test_bingham_fit_sh():
    in_sh = fodf_3x3_order8_descoteaux07.copy()
    max_lobes = 3
    abs_th = 0.0
    rel_th = 0.1
    min_sep_angle = 25
    max_fit_angle = 15
    mask = None
    nbr_processes = 1

    bingham_arr = bingham_fit_sh(in_sh, max_lobes, abs_th, rel_th,
                                 min_sep_angle, max_fit_angle, mask,
                                 nbr_processes)

    assert np.allclose(bingham_arr, fodf_3x3_bingham)


def test_compute_fiber_density():
    bingham = fodf_3x3_bingham.copy()
    m = 50
    mask = None
    nbr_processes = 1

    fd = compute_fiber_density(bingham, m, mask, nbr_processes)
    assert np.allclose(fodf_3x3_bingham_fd, fd)


def test_compute_fiber_spread():
    fd = fodf_3x3_bingham_fd.copy()
    bingham = fodf_3x3_bingham.copy()

    fs = compute_fiber_spread(bingham, fd)
    assert np.allclose(fs, fodf_3x3_bingham_fs)


def test_compute_fiber_fraction():
    fd = fodf_3x3_bingham_fd.copy()

    ff = compute_fiber_fraction(fd)
    assert np.allclose(ff, fodf_3x3_bingham_ff)
