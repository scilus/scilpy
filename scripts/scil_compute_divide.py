#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute microstructure metrics using the DIVIDE method.

By default, will output all possible files, using default names.
Specific names can be specified using the file flags specified in the
"File flags" section.

If --not_all is set, only the files specified explicitly by the flags
will be output.

Based on Markus Nilsson, Filip Szczepankiewicz, Björn Lampinen, André Ahlgren,
João P. de Almeida Martins, Samo Lasic, Carl-Fredrik Westin,
and Daniel Topgaard. An open-source framework for analysis of multidimensional
diffusion MRI data implemented in MATLAB.
Proc. Intl. Soc. Mag. Reson. Med. (26), Paris, France, 2018.
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_force_b0_arg,
                             add_processes_arg)
from scilpy.reconst.multi_processes import fit_gamma
from scilpy.reconst.divide_fit import gamma_fit2metrics
from scilpy.reconst.b_tensor_utils import (generate_powder_averaged_data,
                                           extract_affine)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--in_dwi_linear', metavar='file', default=None,
                   help='Path of the linear input diffusion volume.')
    p.add_argument('--in_bval_linear', metavar='file', default=None,
                   help='Path of the linear bvals file, in FSL format.')
    p.add_argument('--in_bvec_linear', metavar='file', default=None,
                   help='Path of the linear bvecs file, in FSL format.')
    p.add_argument('--in_dwi_planar', metavar='file', default=None,
                   help='Path of the planar input diffusion volume.')
    p.add_argument('--in_bval_planar', metavar='file', default=None,
                   help='Path of the planar bvals file, in FSL format.')
    p.add_argument('--in_bvec_planar', metavar='file', default=None,
                   help='Path of the planar bvecs file, in FSL format.')
    p.add_argument('--in_dwi_spherical', metavar='file', default=None,
                   help='Path of the spherical input diffusion volume.')
    p.add_argument('--in_bval_spherical', metavar='file', default=None,
                   help='Path of the spherical bvals file, in FSL format.')
    p.add_argument('--in_bvec_spherical', metavar='file', default=None,
                   help='Path of the spherical bvecs file, in FSL format.')
    p.add_argument('--in_dwi_custom', metavar='file', default=None,
                   help='Path of the custom input diffusion volume.')
    p.add_argument('--in_bval_custom', metavar='file', default=None,
                   help='Path of the custom bvals file, in FSL format.')
    p.add_argument('--in_bvec_custom', metavar='file', default=None,
                   help='Path of the custom bvecs file, in FSL format.')
    p.add_argument('--in_bdelta_custom', type=float, choices=[0, 1, -0.5, 0.5],
                   help='Value of the b_delta for the custom encoding.')

    p.add_argument(
        '--mask',
        help='Path to a binary mask. Only the data inside the '
             'mask will be used for computations and reconstruction.')
    p.add_argument(
        '--fa',
        help='Path to a FA map. Needed for calculating the OP.')
    p.add_argument(
        '--tolerance', type=int, default=20,
        help='The tolerated gap between the b-values to '
             'extract\nand the current b-value. [%(default)s]')
    p.add_argument(
        '--fit_iters', type=int, default=1,
        help='The number of time the gamma fit will be done [%(default)s]')
    p.add_argument(
        '--random_iters', type=int, default=50,
        help='The number of iterations for the initial parameters search. '
             '[%(default)s]')
    p.add_argument(
        '--do_weight_bvals', action='store_false',
        help='If set, does not do a weighting on the bvalues in the gamma '
             'fit.')
    p.add_argument(
        '--do_weight_pa', action='store_false',
        help='If set, does not do a powder averaging weighting in the gamma '
             'fit.')
    p.add_argument(
        '--redo_weight_bvals', action='store_false',
        help='If set, does not do a second gamma fit with a weighting on the '
             'bvalues using the newly found MD.')
    p.add_argument(
        '--do_multiple_s0', action='store_false',
        help='If set, does not take into account multiple baseline signals.')

    add_force_b0_arg(p)
    add_processes_arg(p)
    add_overwrite_arg(p)

    p.add_argument(
        '--not_all', action='store_true',
        help='If set, only saves the files specified using the '
             'file flags. (Default: False)')

    g = p.add_argument_group(title='File flags')

    g.add_argument(
        '--md', metavar='file', default='',
        help='Output filename for the MD.')
    g.add_argument(
        '--ufa', metavar='file', default='',
        help='Output filename for the microscopic FA.')
    g.add_argument(
        '--op', metavar='file', default='',
        help='Output filename for the order parameter.')
    g.add_argument(
        '--mk_i', metavar='file', default='',
        help='Output filename for the isotropic mean kurtosis.')
    g.add_argument(
        '--mk_a', metavar='file', default='',
        help='Output filename for the anisotropic mean kurtosis.')
    g.add_argument(
        '--mk_t', metavar='file', default='',
        help='Output filename for the total mean kurtosis.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not args.not_all:
        args.md = args.md or 'md.nii.gz'
        args.ufa = args.ufa or 'ufa.nii.gz'
        args.op = args.op or 'op.nii.gz'
        args.mk_i = args.mk_i or 'mk_i.nii.gz'
        args.mk_a = args.mk_a or 'mk_a.nii.gz'
        args.mk_t = args.mk_t or 'mk_t.nii.gz'

    arglist = [args.md, args.ufa, args.op, args.mk_i, args.mk_a, args.mk_t]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least ' +
                     'one file to output.')

    assert_inputs_exist(parser, [],
                        optional=[args.in_dwi_linear, args.in_bval_linear,
                                  args.in_bvec_linear,
                                  args.in_dwi_planar, args.in_bval_planar,
                                  args.in_bvec_planar,
                                  args.in_dwi_spherical,
                                  args.in_bval_spherical,
                                  args.in_bvec_spherical])
    assert_outputs_exist(parser, args, arglist)

    input_files = [args.in_dwi_linear, args.in_dwi_planar,
                   args.in_dwi_spherical, args.in_dwi_custom]
    bvals_files = [args.in_bval_linear, args.in_bval_planar,
                   args.in_bval_spherical, args.in_bval_custom]
    bvecs_files = [args.in_bvec_linear, args.in_bvec_planar,
                   args.in_bvec_spherical, args.in_bvec_custom]
    b_deltas_list = [1.0, -0.5, 0, args.in_bdelta_custom]

    for i in range(4):
        enc = ["linear", "planar", "spherical", "custom"]
        if (input_files[i] is None and bvals_files[i] is None
           and bvecs_files[i] is None):
            inclusive = 1
            if i == 3 and args.in_bdelta_custom is not None:
                inclusive = 0
        elif (input_files[i] is not None and bvals_files[i] is not None
              and bvecs_files[i] is not None):
            inclusive = 1
            if i == 3 and args.in_bdelta_custom is None:
                inclusive = 0
        else:
            inclusive = 0
        if inclusive == 0:
            msg = """All of in_dwi, bval and bvec files are mutually needed
                  for {} encoding."""
            raise ValueError(msg.format(enc[i]))

    tol = args.tolerance
    force_b0_thr = args.force_b0_threshold

    data, gtab_infos = generate_powder_averaged_data(input_files,
                                                     bvals_files,
                                                     bvecs_files,
                                                     b_deltas_list,
                                                     force_b0_thr,
                                                     tol=tol)

    affine = extract_affine(input_files)

    gtab_infos[0] *= 1e6  # getting bvalues to SI units

    if args.mask is None:
        mask = None
    else:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")

    if args.fa is not None:
        vol = nib.load(args.fa)
        FA = vol.get_fdata(dtype=np.float32)

    parameters = fit_gamma(data, gtab_infos, mask=mask,
                           fit_iters=args.fit_iters,
                           random_iters=args.random_iters,
                           do_weight_bvals=args.do_weight_bvals,
                           do_weight_pa=args.do_weight_pa,
                           redo_weight_bvals=args.redo_weight_bvals,
                           do_multiple_s0=args.do_multiple_s0,
                           nbr_processes=args.nbr_processes)

    microFA, MK_I, MK_A, MK_T = gamma_fit2metrics(parameters)
    microFA[np.isnan(microFA)] = 0
    microFA = np.clip(microFA, 0, 1)

    if args.md:
        nib.save(nib.Nifti1Image(parameters[..., 1].astype(np.float32),
                                 affine), args.md)

    if args.ufa:
        nib.save(nib.Nifti1Image(microFA.astype(np.float32), affine), args.ufa)
    if args.op:
        if args.fa is not None:
            OP = np.sqrt((3 * (microFA ** (-2)) - 2) / (3 * (FA ** (-2)) - 2))
            OP[microFA < FA] = 0
            nib.save(nib.Nifti1Image(OP.astype(np.float32), affine), args.op)
        else:
            logging.warning('The FA must be given in order to compute the OP.')
    if args.mk_i:
        nib.save(nib.Nifti1Image(MK_I.astype(np.float32), affine), args.mk_i)
    if args.mk_a:
        nib.save(nib.Nifti1Image(MK_A.astype(np.float32), affine), args.mk_a)
    if args.mk_t:
        nib.save(nib.Nifti1Image(MK_T.astype(np.float32), affine), args.mk_t)


if __name__ == "__main__":
    main()
