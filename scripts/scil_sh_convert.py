#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a SH file between the two of the following basis choices:
'descoteaux07', 'descoteaux07_legacy', 'tournier07' or 'tournier07_legacy'.
Using the sh_basis argument, both the input and the output SH bases must be
given, in the order. For more information about the bases, see
https://docs.dipy.org/stable/theory/sh_basis.html.

Formerly: scil_convert_sh_basis.py
"""

import argparse
import logging

from dipy.data import get_sphere
import nibabel as nib
import numpy as np

from scilpy.reconst.sh import convert_sh_basis
from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             add_processes_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             parse_sh_basis_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_sh',
                   help='Input SH filename. (nii or nii.gz)')
    p.add_argument('out_sh',
                   help='Output SH filename. (nii or nii.gz)')

    add_sh_basis_args(p, mandatory=True, input_output=True)
    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_sh)
    assert_outputs_exist(parser, args, args.out_sh)

    sphere = get_sphere('repulsion724').subdivide(1)
    img = nib.load(args.in_sh)
    data = img.get_fdata(dtype=np.float32)

    in_sh_basis, is_in_legacy, out_sh_basis, is_out_legacy \
        = parse_sh_basis_arg(args)

    new_data = convert_sh_basis(data, sphere,
                                input_basis=in_sh_basis,
                                output_basis=out_sh_basis,
                                is_input_legacy=is_in_legacy,
                                is_output_legacy=is_out_legacy,
                                nbr_processes=args.nbr_processes)

    nib.save(nib.Nifti1Image(new_data, img.affine, header=img.header),
             args.out_sh)


if __name__ == "__main__":
    main()
