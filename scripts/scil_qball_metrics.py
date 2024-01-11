#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute the Constant Solid Angle (CSA) or Analytical Q-ball model,
the generalized fractional anisotropy (GFA) and the peaks of the model.

By default, will output all possible files, using default names. Specific names
can be specified using the file flags specified in the "File flags" section.

If --not_all is set, only the files specified explicitly by the flags
will be output.

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
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_sh_basis_args, assert_inputs_exist,
                             assert_outputs_exist, add_force_b0_arg,
                             validate_nbr_processes, add_verbose_arg)
from scilpy.io.image import get_data_as_mask
from scilpy.gradients.bvec_bval_tools import (normalize_bvecs,
                                              is_normalized_bvecs,
                                              check_b0_threshold)


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
    g.add_argument('--gfa', default='',
                   help='Output filename for the generalized fractional '
                        'anisotropy [gfa.nii.gz].')
    g.add_argument('--peaks', default='',
                   help='Output filename for the extracted peaks '
                        '[peaks.nii.gz].')
    g.add_argument('--peak_indices', default='',
                   help='Output filename for the generated peaks '
                        'indices on the sphere [peaks_indices.nii.gz].')
    g.add_argument('--sh', default='',
                   help='Output filename for the spherical harmonics '
                        'coefficients [sh.nii.gz].')
    g.add_argument('--nufo', default='',
                   help='Output filename for the NUFO map [nufo.nii.gz].')
    g.add_argument('--a_power', default='',
                   help='Output filename for the anisotropic power map'
                        '[anisotropic_power.nii.gz].')

    add_sh_basis_args(p)
    add_processes_arg(p)
    add_verbose_arg(p)
    add_force_b0_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

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
        parser.error('When using --not_all, you need to specify at least ' +
                     'one file to output.')

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec])
    assert_outputs_exist(parser, args, arglist)
    validate_nbr_processes(parser, args)

    nbr_processes = args.nbr_processes
    parallel = nbr_processes > 1

    # Load data
    img = nib.load(args.in_dwi)
    data = img.get_fdata(dtype=np.float32)

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)

    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    check_b0_threshold(args, bvals.min())
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    sphere = get_sphere('symmetric724')

    mask = None
    if args.mask:
        mask = get_data_as_mask(nib.load(args.mask))

        # Sanity check on shape of mask
        if mask.shape != data.shape[:-1]:
            raise ValueError('Mask shape does not match data shape.')

    if args.use_qball:
        model = QballModel(gtab, sh_order=args.sh_order,
                           smooth=DEFAULT_SMOOTH)
    else:
        model = CsaOdfModel(gtab, sh_order=args.sh_order,
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
                                sh_order=int(args.sh_order),
                                sh_basis_type=args.sh_basis,
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
