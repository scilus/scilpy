#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Convert a SH file between the two commonly used bases
    ('descoteaux07' or 'tournier07'). The specified basis corresponds to the
    input data basis.
"""

import argparse

from dipy.data import get_sphere
import nibabel as nib
import numpy as np

from scilpy.reconst.multi_processes import convert_sh_basis
from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             add_processes_arg,
                             assert_inputs_exist, assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_sh',
                   help='Input SH filename. (nii or nii.gz)')

    p.add_argument('out_sh',
                   help='Output SH filename. (nii or nii.gz)')

    add_sh_basis_args(p, mandatory=True)
    add_processes_arg(p)
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
                                nbr_processes=args.nbr_processes)

    nib.save(nib.Nifti1Image(new_data, img.affine, header=img.header),
             args.out_sh)


if __name__ == "__main__":
    main()
