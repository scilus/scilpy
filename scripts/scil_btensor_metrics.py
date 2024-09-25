#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute microstructure metrics using the DIVIDE method. In order to
operate, the script needs at leats two different types of b-tensor encodings.
Note that custom encodings are not yet supported, so that only the linear
tensor encoding (LTE, b_delta = 1), the planar tensor encoding
(PTE, b_delta = -0.5), the spherical tensor encoding (STE, b_delta = 0) and
the cigar shape tensor encoding (b_delta = 0.5) are available. Moreover, all
of `--in_dwis`, `--in_bvals`, `--in_bvecs` and `--in_bdeltas` must have the
same number of arguments. Be sure to keep the same order of encodings
throughout all these inputs and to set `--in_bdeltas` accordingly (IMPORTANT).

By default, will output all possible files, using default names. Thus, this
script outputs the results from the DIVIDE fit or direct derivatives:
mean diffusivity (MD), isotropic mean kurtosis (mk_i), anisotropic mean
kurtosis (mk_a), total mean kurtosis (mk_t) and finally micro-FA (uFA).
Specific names can be specified using the
file flags specified in the "File flags" section.

If --not_all is set, only the files specified explicitly by the flags
will be output. The order parameter can also be computed from the uFA and a
precomputed FA, using separate input parameters.

>>> scil_btensor_metrics.py --in_dwis LTE.nii.gz PTE.nii.gz STE.nii.gz
    --in_bvals LTE.bval PTE.bval STE.bval --in_bvecs LTE.bvec PTE.bvec STE.bvec
    --in_bdeltas 1 -0.5 0 --mask mask.nii.gz

IMPORTANT: If the script does not converge to a solution, it is probably due to
noise outside the brain. Thus, it is strongly recommanded to provide a brain
mask with --mask.

Based on Markus Nilsson, Filip Szczepankiewicz, Björn Lampinen, André Ahlgren,
João P. de Almeida Martins, Samo Lasic, Carl-Fredrik Westin,
and Daniel Topgaard. An open-source framework for analysis of multidimensional
diffusion MRI data implemented in MATLAB.
Proc. Intl. Soc. Mag. Reson. Med. (26), Paris, France, 2018.

Formerly: scil_compute_divide.py
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.image.utils import extract_affine
from scilpy.io.btensor import generate_btensor_input
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_processes_arg,
                             add_verbose_arg, add_skip_b0_check_arg,
                             add_tolerance_arg, assert_headers_compatible)
from scilpy.reconst.divide import fit_gamma, gamma_fit2metrics


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--in_dwis', nargs='+', required=True,
                   help='Path to the input diffusion volume for each '
                        'b-tensor encoding type.')
    p.add_argument('--in_bvals', nargs='+', required=True,
                   help='Path to the bval file, in FSL format, for each '
                        'b-tensor encoding type.')
    p.add_argument('--in_bvecs', nargs='+', required=True,
                   help='Path to the bvec file, in FSL format, for each '
                        'b-tensor encoding type.')
    p.add_argument('--in_bdeltas', nargs='+', type=float,
                   choices=[0, 1, -0.5, 0.5], required=True,
                   help='Value of b_delta for each b-tensor encoding type, '
                        'in the same order as dwi, bval and bvec inputs.')

    p.add_argument(
        '--mask',
        help='Path to a binary mask. Only the data inside the '
             'mask will be used for computations and reconstruction.')
    add_tolerance_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=False,
                          b0_tol_name='--tolerance')
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
        '--do_multiple_s0', action='store_false',
        help='If set, does not take into account multiple baseline signals.')

    g2 = p.add_argument_group(title='Order parameter (OP)')
    g2.add_argument(
        '--op',
        help='Output filename for the order parameter. The OP will not be '
             'output if this is not given. Computation of the OP also '
             'requires a precomputed FA map (given using --fa).')
    g2.add_argument(
        '--fa',
        help='Path to a FA map. Needed for calculating the OP.')

    add_processes_arg(p)
    add_verbose_arg(p)
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
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    if not args.not_all:
        args.md = args.md or 'md.nii.gz'
        args.ufa = args.ufa or 'ufa.nii.gz'
        args.mk_i = args.mk_i or 'mk_i.nii.gz'
        args.mk_a = args.mk_a or 'mk_a.nii.gz'
        args.mk_t = args.mk_t or 'mk_t.nii.gz'

    arglist = [args.md, args.ufa, args.mk_i, args.mk_a, args.mk_t]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least '
                     'one file to output.')

    assert_inputs_exist(parser, args.in_dwis + args.in_bvals + args.in_bvecs,
                        [args.mask, args.fa])
    assert_outputs_exist(parser, args, arglist)
    assert_headers_compatible(parser, args.in_dwis, [args.mask, args.fa])

    if args.op and not args.fa:
        parser.error('Computation of the OP requires a precomputed '
                     'FA map (given using --fa).')

    if not (len(args.in_dwis) == len(args.in_bvals)
            == len(args.in_bvecs) == len(args.in_bdeltas)):
        msg = """The number of given dwis, bvals, bvecs and bdeltas must be the
              same. Please verify that all inputs were correctly inserted."""
        raise ValueError(msg)

    if len(np.unique(args.in_bdeltas)) < 2:
        msg = """At least two different b-tensor shapes are needed for the
              script to work properly."""
        raise ValueError(msg)

    # Loading
    affine = extract_affine(args.in_dwis)

    # Note. This script does not currently allow using a separate b0_threshold
    # for the b0s. Using the tolerance. To change this, we would have to
    # change generate_btensor_input.
    data, gtab_infos = generate_btensor_input(args.in_dwis,
                                              args.in_bvals,
                                              args.in_bvecs,
                                              args.in_bdeltas,
                                              do_pa_signals=True,
                                              tol=args.tolerance,
                                              skip_b0_check=args.skip_b0_check)

    gtab_infos[0] *= 1e6  # getting bvalues to SI units

    if args.mask is None:
        mask = None
        logging.warning(
            'No mask provided. The fit might not converge due to noise. '
            'Please provide a mask if it is the case.')
    else:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)

    if args.fa is not None:
        vol = nib.load(args.fa)
        FA = vol.get_fdata(dtype=np.float32)

    # Processing
    parameters = fit_gamma(data, gtab_infos, mask=mask,
                           fit_iters=args.fit_iters,
                           random_iters=args.random_iters,
                           do_weight_bvals=args.do_weight_bvals,
                           do_weight_pa=args.do_weight_pa,
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
        OP = np.sqrt((3 * (microFA ** (-2)) - 2) / (3 * (FA ** (-2)) - 2))
        OP[microFA < FA] = 0
        nib.save(nib.Nifti1Image(OP.astype(np.float32), affine), args.op)
    if args.mk_i:
        nib.save(nib.Nifti1Image(MK_I.astype(np.float32), affine), args.mk_i)
    if args.mk_a:
        nib.save(nib.Nifti1Image(MK_A.astype(np.float32), affine), args.mk_a)
    if args.mk_t:
        nib.save(nib.Nifti1Image(MK_T.astype(np.float32), affine), args.mk_t)


if __name__ == "__main__":
    main()
