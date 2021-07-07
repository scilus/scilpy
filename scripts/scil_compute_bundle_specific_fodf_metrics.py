#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import argparse

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.reconst.bingham import (bingham_fit_sh_parallel,
                                    compute_fiber_density_parallel,
                                    compute_fiber_spread,
                                    compute_structural_complexity)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('in_sh', help='Input SH image.')

    p.add_argument('--out_bingham', default='bingham.nii.gz',
                   help='Output bingham functions image.')
    p.add_argument('--out_fd', default='fd.nii.gz',
                   help='Path to output fiber density [%(default)s].')
    p.add_argument('--out_fs', default='fs.nii.gz',
                   help='Path to output fiber spread [%(default)s].')
    p.add_argument('--out_cx', default='cx.nii.gz',
                   help='Path to structural complexity file [%(default)s].')

    p.add_argument('--max_lobes', type=int, default=5,
                   help='Maximum number of lobes per voxel'
                        ' to extract [%(default)s].')
    p.add_argument('--a_th', type=float, default=0.0,
                   help='Absolute threshold for peaks'
                        ' extraction [%(default)s].')
    p.add_argument('--r_th', type=float, default=0.2,
                   help='Relative threshold for peaks'
                        ' extraction [%(default)s].')
    p.add_argument('--min_sep_angle', type=float, default=25,
                   help='Minimum separation angle between'
                        ' two peaks [%(default)s].')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    outputs = [args.out_bingham, args.out_fd, args.out_fs, args.out_cx]
    assert_inputs_exist(parser, args.in_sh)
    assert_outputs_exist(parser, args, outputs)

    sh_im = nib.load(args.in_sh)
    data = sh_im.get_fdata()

    bingham = bingham_fit_sh_parallel(data, args.max_lobes)
    nib.save(nib.Nifti1Image(bingham, sh_im.affine), args.out_bingham)

    fd = compute_fiber_density_parallel(bingham)
    nib.save(nib.Nifti1Image(fd, sh_im.affine), args.out_fd)

    fs = compute_fiber_spread(bingham, fd)
    nib.save(nib.Nifti1Image(fs, sh_im.affine), args.out_fs)

    cx = compute_structural_complexity(fd)
    nib.save(nib.Nifti1Image(cx, sh_im.affine), args.out_cx)


if __name__ == '__main__':
    main()
