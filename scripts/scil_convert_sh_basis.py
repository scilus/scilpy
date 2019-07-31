#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from dipy.data import get_sphere
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv
import nibabel as nib
import numpy as np

from scilpy.reconst.utils import find_order_from_nb_coeff
from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             assert_inputs_exist, assert_outputs_exists)

DESCRIPTION = """
    Convert a SH file between the two commonly used bases
    ('descoteaux07' or 'tournier07'). The specified basis corresponds to the
    input data basis.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=DESCRIPTION)

    p.add_argument('input_sh',
                   help='Input SH filename (nii or nii.gz)')

    p.add_argument('output_name',
                   help='Name of the output file')

    add_sh_basis_args(p, mandatory=True)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.input_sh])
    assert_outputs_exists(parser, args, [args.output_name])

    input_basis = args.sh_basis
    output_basis = 'descoteaux07' if input_basis == 'tournier07' else 'tournier07'

    sph_harm_basis_ori = sph_harm_lookup.get(input_basis)
    sph_harm_basis_des = sph_harm_lookup.get(output_basis)

    sphere = get_sphere('repulsion724').subdivide(1)
    img = nib.load(args.input_sh)
    data = img.get_data()
    sh_order = find_order_from_nb_coeff(data)

    b_ori, m_ori, n_ori = sph_harm_basis_ori(sh_order, sphere.theta,
                                             sphere.phi)
    b_des, m_des, n_des = sph_harm_basis_des(sh_order, sphere.theta,
                                             sphere.phi)
    l_des = -n_des * (n_des + 1)
    inv_b_des = smooth_pinv(b_des, 0 * l_des)

    indices = np.argwhere(np.any(data, axis=3))
    for i, ind in enumerate(indices):
        ind = tuple(ind)
        sf_1 = np.dot(data[ind], b_ori.T)
        data[ind] = np.dot(sf_1, inv_b_des.T)

    img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), args.output_name)


if __name__ == "__main__":
    main()
