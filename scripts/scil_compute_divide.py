#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
TOTO
"""

import argparse
import logging

from dipy.core.gradients import GradientTable
from dipy.data import get_sphere, default_sphere
from dipy.reconst import shm
from dipy.reconst.mcsd import MultiShellResponse, MultiShellDeconvModel
from dipy.sims.voxel import single_tensor
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_force_b0_arg,
                             add_sh_basis_args, add_processes_arg)
from scilpy.reconst.multi_processes.py import fit_gamma
from scilpy.reconst.divide_fit import gamma_fit2metrics
from scilpy.reconst.b_tensor_utils import generate_powder_averaged_data, extract_affine


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument(
        '--input_linear', metavar='file', default=None,
        help='Path of the linear input diffusion volume.')
    p.add_argument(
        '--bvals_linear', metavar='file', default=None,
        help='Path of the linear bvals file, in FSL format.')
    p.add_argument(
        '--bvecs_linear', metavar='file', default=None,
        help='Path of the linear bvecs file, in FSL format.')
    p.add_argument(
        '--input_planar', metavar='file', default=None,
        help='Path of the planar input diffusion volume.')
    p.add_argument(
        '--bvals_planar', metavar='file', default=None,
        help='Path of the planar bvals file, in FSL format.')
    p.add_argument(
        '--bvecs_planar', metavar='file', default=None,
        help='Path of the planar bvecs file, in FSL format.')
    p.add_argument(
        '--input_spherical', metavar='file', default=None,
        help='Path of the spherical input diffusion volume.')
    p.add_argument(
        '--bvals_spherical', metavar='file', default=None,
        help='Path of the spherical bvals file, in FSL format.')
    p.add_argument(
        '--bvecs_spherical', metavar='file', default=None,
        help='Path of the spherical bvecs file, in FSL format.')
    p.add_argument(
        '--input_custom', metavar='file', default=None,
        help='Path of the custom input diffusion volume.')
    p.add_argument(
        '--bvals_custom', metavar='file', default=None,
        help='Path of the custom bvals file, in FSL format.')
    p.add_argument(
        '--bvecs_custom', metavar='file', default=None,
        help='Path of the custom bvecs file, in FSL format.')
    p.add_argument(
        '--bdelta_custom', type=float, choices=[0, 1, -0.5, 0.5],
        help='Value of the b_delta for the custom encoding.')

    p.add_argument(
        '--mask',
        help='Path to a binary mask. Only the data inside the '
             'mask will be used for computations and reconstruction.')
    p.add_argument(
        '--fa',
        help='Path to a FA map.')
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
        '--do_weight_bvals', action='store_true',
        help='If set, does a weighting on the bvalues in the gamma fit.')
    p.add_argument(
        '--do_weight_pa', action='store_true',
        help='If set, does a powder averaging weighting in the gamma fit.')
    p.add_argument(
        '--redo_weight_bvals', action='store_true',
        help='If set, does a second gamma fit with a weighting on the bvalues '
             'using the newly found MD.')
    p.add_argument(
        '--do_multiple_s0', action='store_true',
        help='If set, takes into account multiple baseline signals.')

    p.add_argument(
        '--not_all', action='store_true',
        help='If set, only saves the files specified using the '
             'file flags. (Default: False)')

    add_force_b0_arg(p)
    add_processes_arg(p)

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

    add_overwrite_arg(p)

    return p

def main():
    parser = _build_arg_parser() # !!!!!!!!!!!!!!!!!!!!!!!!! Add to parser : all linear input mandatory if input_linear is given (for example)
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
                        optional=[args.input_linear, args.bvals_linear,
                                  args.bvecs_linear, args.input_planar,
                                  args.bvals_planar, args.bvecs_planar,
                                  args.input_spherical, args.bvals_spherical,
                                  args.bvecs_spherical])
    assert_outputs_exist(parser, args, arglist)

    # Loading data
    input_files = [args.input_linear, args.input_planar,
                            args.input_spherical, args.input_custom]
    bvals_files = [args.bvals_linear, args.bvals_planar,
                           args.bvals_spherical, args.bvals_custom]
    bvecs_files = [args.bvecs_linear, args.bvecs_planar, 
                           args.bvecs_spherical, args.bvecs_custom]
    b_deltas_list = [1.0, -0.5, 0, args.bdelta_custom]

    data, gtab_infos = generate_powder_averaged_data(input_files,
                                                     bvals_files,
                                                     bvecs_files,
                                                     b_deltas_list,
                                                     args.force_b0_threshold,
                                                     tol=args.tolerance)

    affine = extract_affine(input_files)

    gtab_infos[0] *= 1e6 # getting bvalues to SI units

    if args.mask is None:
        mask = None
    else:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")

    if args.fa is not None:
        vol = nib.load(args.fa)
        FA = vol.get_fdata(dtype=np.float32)

    #print("Signal input linear: ", data[4,2,5,0:5])
    #print("Signal input spherical: ", data[4,2,5,5:10])
    # print(gtab_infos)
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
    # microFA = np.clip(microFA, 0, 1)
    # print(microFA)

    if args.md:
        nib.save(nib.Nifti1Image(parameters[..., 1].astype(np.float32), affine), args.md)

    if args.ufa:
        nib.save(nib.Nifti1Image(microFA.astype(np.float32), affine), args.ufa)
    if args.op:
        if args.fa is not None:
            OP = np.sqrt((3 * (microFA ** (-2)) -2) / (3 * (FA ** (-2)) - 2))
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
