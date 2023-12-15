#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a SH file between the two commonly used bases
('descoteaux07' or 'tournier07'). The specified basis corresponds to the
input data basis. Note that by default, both legacy 'descoteaux07' and
legacy 'tournier07' bases will be assumed. For more information, see
https://dipy.org/documentation/1.4.0./theory/sh_basis/.

Formerly: scil_convert_sh_basis.py
"""

import argparse

from dipy.data import get_sphere
import nibabel as nib
import numpy as np

from scilpy.reconst.sh import convert_sh_basis
from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             add_processes_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_sh',
                   help='Input SH filename. (nii or nii.gz)')
    p.add_argument('out_sh',
                   help='Output SH filename. (nii or nii.gz)')

    p.add_argument('--in_sh_is_not_legacy', action='store_true',
                   help='If set, this means that the input SH are not encoded '
                        'with the legacy version of their SH basis.')
    p.add_argument('--out_sh_is_not_legacy', action='store_true',
                   help='If set, this means that the output SH will not be '
                        'encoded with the legacy version of their SH basis.')

    add_sh_basis_args(p, mandatory=True)
    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_sh)
    assert_outputs_exist(parser, args, args.out_sh)

    sphere = get_sphere('repulsion724').subdivide(1)
    img = nib.load(args.in_sh)
    data = img.get_fdata(dtype=np.float32)

    new_data = convert_sh_basis(data, sphere,
                                input_basis=args.sh_basis,
                                nbr_processes=args.nbr_processes,
                                is_input_legacy=not args.in_sh_is_not_legacy,
                                is_output_legacy=not args.out_sh_is_not_legacy)

    nib.save(nib.Nifti1Image(new_data, img.affine, header=img.header),
             args.out_sh)


if __name__ == "__main__":
    main()
