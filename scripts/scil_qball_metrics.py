#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute the Constant Solid Angle (CSA) or Analytical Q-ball model,
the generalized fractional anisotropy (GFA) and the peaks of the model.

By default, will output all possible files, using default names. Specific names
can be specified using the file flags specified in the "File flags" section.

If --not_all is set, only the files specified explicitly by the flags will be
output.

See [Descoteaux et al MRM 2007, Aganj et al MRM 2009] for details and
[Cote et al MEDIA 2013] for quantitative comparisons.

Formerly: scil_compute_qball_metrics.py
"""
import argparse
import logging

import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.io import read_bvals_bvecs
from dipy.direction.peaks import (peaks_from_model,
                                  reshape_peaks_for_visualization)
from dipy.reconst.shm import QballModel, CsaOdfModel, anisotropic_power

from scilpy.gradients.bvec_bval_tools import (check_b0_threshold,
                                              is_normalized_bvecs,
                                              normalize_bvecs)
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_b0_thresh_arg, add_overwrite_arg,
                             add_processes_arg, add_sh_basis_args,
                             add_skip_b0_check_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             parse_sh_basis_arg, validate_nbr_processes,
                             assert_headers_compatible)


DEFAULT_SMOOTH = 0.006


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi',
                   help='Path of the input diffusion volume.')
    p.add_argument('in_bval',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('in_bvec',
                   help='Path of the bvecs file, in FSL format.')

    add_overwrite_arg(p)
    p.add_argument('--sh_order', default=4, type=int,
                   help='Spherical harmonics order. Must be a positive even '
                        'number [%(default)s].')
    p.add_argument('--mask',
                   help='Path to a binary mask. Only data inside the mask will'
                        ' be used for computations and reconstruction '
                        '[%(default)s].')
    p.add_argument('--use_qball', action='store_true',
                   help='If set, qball will be used as the odf reconstruction'
                        ' model instead of CSA.')
    p.add_argument('--not_all', action='store_true',
                   help='If set, will only save the files specified using the '
                        'following flags.')

    g = p.add_argument_group(title='File flags')
    g.add_argument('--gfa',
                   help='Output filename for the generalized fractional '
                        'anisotropy [gfa.nii.gz].')
    g.add_argument('--peaks',
                   help='Output filename for the extracted peaks '
                        '[peaks.nii.gz].')
    g.add_argument('--peak_indices',
                   help='Output filename for the generated peaks '
                        'indices on the sphere [peaks_indices.nii.gz].')
    g.add_argument('--sh',
                   help='Output filename for the spherical harmonics '
                        'coefficients [sh.nii.gz].')
    g.add_argument('--nufo',
                   help='Output filename for the NUFO map [nufo.nii.gz].')
    g.add_argument('--a_power',
                   help='Output filename for the anisotropic power map'
                        '[anisotropic_power.nii.gz].')

    add_b0_thresh_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=True)
    add_sh_basis_args(p)
    add_processes_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    if not args.not_all:
        args.gfa = args.gfa or 'gfa.nii.gz'
        args.peaks = args.peaks or 'peaks.nii.gz'
        args.peak_indices = args.peak_indices or 'peaks_indices.nii.gz'
        args.sh = args.sh or 'sh.nii.gz'
        args.nufo = args.nufo or 'nufo.nii.gz'
        args.a_power = args.a_power or 'anisotropic_power.nii.gz'

    arglist = [args.gfa, args.peaks, args.peak_indices, args.sh, args.nufo,
               args.a_power]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least one '
                     'file to output.')

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec],
                        args.mask)
    assert_outputs_exist(parser, args, [], optional=arglist)
    assert_headers_compatible(parser, args.in_dwi, args.mask)

    nbr_processes = validate_nbr_processes(parser, args)
    parallel = nbr_processes > 1

    # Load data
    img = nib.load(args.in_dwi)
    data = img.get_fdata(dtype=np.float32)

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized... Normalizing '
                        'now.')
        bvecs = normalize_bvecs(bvecs)

    # Usage of gtab.b0s_mask in dipy's models is not very well documented, but
    # we can see that it is indeed used.
    args.b0_threshold = check_b0_threshold(bvals.min(),
                                           b0_thr=args.b0_threshold,
                                           skip_b0_check=args.skip_b0_check)
    gtab = gradient_table(bvals, bvecs, b0_threshold=args.b0_threshold)

    sphere = get_sphere('symmetric724')
    sh_basis, is_legacy = parse_sh_basis_arg(args)

    mask = get_data_as_mask(nib.load(args.mask)) if args.mask else None

    if args.use_qball:
        model = QballModel(gtab, sh_order_max=args.sh_order,
                           smooth=DEFAULT_SMOOTH)
    else:
        model = CsaOdfModel(gtab, sh_order_max=args.sh_order,
                            smooth=DEFAULT_SMOOTH)

    odfpeaks = peaks_from_model(model=model,
                                data=data,
                                sphere=sphere,
                                relative_peak_threshold=.5,
                                min_separation_angle=25,
                                mask=mask,
                                return_odf=False,
                                normalize_peaks=True,
                                return_sh=True,
                                sh_order_max=int(args.sh_order),
                                sh_basis_type=sh_basis,
                                legacy=is_legacy,
                                npeaks=5,
                                parallel=parallel,
                                num_processes=nbr_processes)

    if args.gfa:
        nib.save(nib.Nifti1Image(odfpeaks.gfa.astype(np.float32), img.affine),
                 args.gfa)

    if args.peaks:
        nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(odfpeaks),
                 img.affine), args.peaks)

    if args.peak_indices:
        nib.save(nib.Nifti1Image(odfpeaks.peak_indices, img.affine),
                 args.peak_indices)

    if args.sh:
        nib.save(nib.Nifti1Image(
            odfpeaks.shm_coeff.astype(np.float32), img.affine),
            args.sh)

    if args.nufo:
        peaks_count = (odfpeaks.peak_indices > -1).sum(3)
        nib.save(nib.Nifti1Image(peaks_count.astype(np.int32), img.affine),
                 args.nufo)

    if args.a_power:
        odf_a_power = anisotropic_power(odfpeaks.shm_coeff)
        nib.save(nib.Nifti1Image(odf_a_power.astype(np.float32), img.affine),
                 args.a_power)


if __name__ == "__main__":
    main()
