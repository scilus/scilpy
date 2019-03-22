#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import os
import warnings

from dipy.core.sphere import Sphere
from dipy.utils.arrfuncs import as_native_array
from dipy.reconst.peaks import peak_directions
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv
import numpy as np


def find_order_from_nb_coeff(data):
    if isinstance(data, np.ndarray):
        shape = data.shape
    else:
        shape = data
    return int((-3 + np.sqrt(1 + 8 * shape[-1])) / 2)


def _honor_authorsnames_sh_basis(sh_basis_type):
    sh_basis = sh_basis_type
    if sh_basis_type == 'fibernav':
        sh_basis = 'descoteaux07'
        warnings.warn("'fibernav' sph basis name is deprecated and will be "
                      "discontinued in favor of 'descoteaux07'.", DeprecationWarning)
    elif sh_basis_type == 'mrtrix':
        sh_basis = 'tournier07'
        warnings.warn("'mrtrix' sph basis name is deprecated and will be "
                      "discontinued in favor of 'tournier07'.", DeprecationWarning)
    return sh_basis


def get_b_matrix(order, sphere, sh_basis_type, return_all=False):
    sh_basis = _honor_authorsnames_sh_basis(sh_basis_type)
    sph_harm_basis = sph_harm_lookup.get(sh_basis)
    if sph_harm_basis is None:
        raise ValueError("Invalid basis name.")
    b_matrix, m, n = sph_harm_basis(order, sphere.theta, sphere.phi)
    if return_all:
        return b_matrix, m, n
    return b_matrix
