#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute the maximum Apparent Fiber Density (AFD), the fiber ODFs
orientations, values and indices (peaks, peak_values, peak_indices), the Number
of Fiber Orientations (NuFO) maps from fiber ODFs and the RGB map.

AFD_max map is the maximal fODF amplitude for each voxel.

NuFO is the the number of maxima of the fODF with an ABSOLUTE amplitude above
the threshold set using --at, AND an amplitude above the RELATIVE threshold
set using --rt.

The --at argument should be set to a value which is 1.5 times the maximal
value of the fODF in the ventricules. This can be obtained with the
compute_fodf_max_in_ventricules.py script.

By default, will output all possible files, using default names. Specific names
can be specified using the file flags specified in the "File flags" section.

If --not_all is set, only the files specified explicitly by the flags will be
output.

See [Raffelt et al. NeuroImage 2012] and [Dell'Acqua et al HBM 2013] for the
definitions.
"""

import argparse
import numpy as np
import nibabel as nib

from dipy.data import get_sphere
from dipy.direction.peaks import reshape_peaks_for_visualization

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             add_processes_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.reconst.multi_processes import peaks_from_sh, maps_from_sh


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_fODF',
                   help='Path of the fODF volume in spherical harmonics (SH).')

    p.add_argument('--sphere', metavar='string', default='repulsion724',
                   choices=['repulsion100', 'repulsion724'],
                   help='Discrete sphere to use in the processing '
                        '[%(default)s].')
    p.add_argument('--mask', metavar='',
                   help='Path to a binary mask. Only the data inside the mask\n'
                        'will beused for computations and reconstruction '
                        '[%(default)s].')
    p.add_argument('--at', dest='a_threshold', type=float, default='0.0',
                   help='Absolute threshold on fODF amplitude. This '
                        'value should be set to\napproximately 1.5 to 2 times '
                        'the maximum fODF amplitude in isotropic voxels\n'
                        '(ie. ventricles).\nUse compute_fodf_max_in_ventricles.py '
                        'to find the maximal value.\n'
                        'See [Dell\'Acqua et al HBM 2013] [%(default)s].')
    p.add_argument('--rt', dest='r_threshold', type=float, default='0.1',
                   help='Relative threshold on fODF amplitude in percentage  '
                        '[%(default)s].')
    add_sh_basis_args(p)
    add_overwrite_arg(p)
    add_processes_arg(p)
    p.add_argument('--not_all', action='store_true',
                   help='If set, only saves the files specified using the '
                        'file flags [%(default)s].')

    g = p.add_argument_group(title='File flags')
    g.add_argument('--afd_max', metavar='file', default='',
                   help='Output filename for the AFD_max map.')
    g.add_argument('--afd_total', metavar='file', default='',
                   help='Output filename for the AFD_total map (SH coeff = 0).')
    g.add_argument('--afd_sum', metavar='file', default='',
                   help='Output filename for the sum of all peak contributions\n'
                        '(sum of fODF lobes on the sphere).')
    g.add_argument('--nufo', metavar='file', default='',
                   help='Output filename for the NuFO map.')
    g.add_argument('--rgb', metavar='file', default='',
                   help='Output filename for the RGB map.')
    g.add_argument('--peaks', metavar='file', default='',
                   help='Output filename for the extracted peaks.')
    g.add_argument('--peak_values', metavar='file', default='',
                   help='Output filename for the extracted peaks values.')
    g.add_argument('--peak_indices', metavar='file', default='',
                   help='Output filename for the generated peaks indices on '
                        'the sphere.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.not_all:
        args.afd_max = args.afd_max or 'afd_max.nii.gz'
        args.afd_total = args.afd_total or 'afd_total_sh0.nii.gz'
        args.afd_sum = args.afd_sum or 'afd_sum.nii.gz'
        args.nufo = args.nufo or 'nufo.nii.gz'
        args.rgb = args.rgb or 'rgb.nii.gz'
        args.peaks = args.peaks or 'peaks.nii.gz'
        args.peak_values = args.peak_values or 'peak_values.nii.gz'
        args.peak_indices = args.peak_indices or 'peak_indices.nii.gz'

    arglist = [args.afd_max, args.afd_total, args.afd_sum, args.nufo,
               args.rgb, args.peaks, args.peak_values,
               args.peak_indices]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least '
                     'one file to output.')

    assert_inputs_exist(parser, args.in_fODF)
    assert_outputs_exist(parser, args, arglist)

    vol = nib.load(args.in_fODF)
    data = vol.get_fdata(dtype=np.float32)
    affine = vol.affine

    if args.mask is None:
        mask = None
    else:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")

    sphere = get_sphere(args.sphere)

    # Computing peaks
    peak_dirs, peak_values, \
        peak_indices = peaks_from_sh(data,
                                     sphere,
                                     mask=mask,
                                     relative_peak_threshold=args.r_threshold,
                                     absolute_threshold=args.a_threshold,
                                     min_separation_angle=25,
                                     normalize_peaks=False,
                                     sh_basis_type=args.sh_basis,
                                     nbr_processes=args.nbr_processes)

    # Computing maps
    if args.nufo or args.afd_max or args.afd_total or args.afd_sum or args.rgb:
        nufo_map, afd_max, afd_sum, rgb_map, \
            _, _ = maps_from_sh(data, peak_dirs, peak_values, peak_indices,
                                sphere, nbr_processes=args.nbr_processes)

    # Save result
    if args.nufo:
        nib.save(nib.Nifti1Image(nufo_map.astype(np.float32),
                                 affine), args.nufo)

    if args.afd_max:
        nib.save(nib.Nifti1Image(afd_max.astype(np.float32),
                                 affine), args.afd_max)

    if args.afd_total:
        # this is the analytical afd total
        afd_tot = data[:, :, :, 0]
        nib.save(nib.Nifti1Image(afd_tot.astype(np.float32),
                                 affine), args.afd_total)

    if args.afd_sum:
        nib.save(nib.Nifti1Image(afd_sum.astype(np.float32),
                                 affine), args.afd_sum)

    if args.rgb:
        nib.save(nib.Nifti1Image(rgb_map.astype('uint8'), affine), args.rgb)

    if args.peaks or args.peak_values:
        peak_values = np.divide(peak_values, peak_values[..., 0, None],
                                out=np.zeros_like(peak_values),
                                where=peak_values[..., 0, None] != 0)
        peak_dirs[...] *= peak_values[..., :, None]
        if args.peaks:
            nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peak_dirs),
                                     affine), args.peaks)
        if args.peak_values:
            nib.save(nib.Nifti1Image(peak_values, vol.affine), args.peak_values)

    if args.peak_indices:
        nib.save(nib.Nifti1Image(peak_indices, vol.affine), args.peak_indices)


if __name__ == "__main__":
    main()
