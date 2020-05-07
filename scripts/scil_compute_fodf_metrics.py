#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute the maximum Apparent Fiber Density (AFD), the fiber ODFs
orientations (peaks) and the Number of Fiber Orientations (NuFO) maps from
fiber ODFs.

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
import os
import numpy as np
import nibabel as nib

from dipy.core.ndindex import ndindex
from dipy.data import get_sphere
from dipy.direction.peaks import reshape_peaks_for_visualization

from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.reconst.utils import (
    find_order_from_nb_coeff, get_b_matrix, get_maximas)
from scilpy.reconst.multi_process import peaks_from_sh, maps_from_sh


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'input', metavar='fODFs',
        help='Path of the fODF volume in spherical harmonics (SH).')
    p.add_argument(
        'at', metavar='a_threshold', type=float,
        help='WARNING!!! EXTREMELY IMPORTANT PARAMETER, VARIABLE '
             'ACROSS DATASETS!!!\nAbsolute threshold on fODF amplitude.\nThis '
             'value should set to approximately 1.5 to 2 times the maximum\n'
             'fODF amplitude in isotropic voxels (ex. ventricles).\n'
             'compute_fodf_max_in_ventricles.py can be used to find the '
             'maximal value.\nSee [Dell\'Acqua et al HBM 2013].')

    p.add_argument(
        '--sphere', metavar='string', default='repulsion724',
        choices=['repulsion100', 'repulsion724'],
        help='Discrete sphere to use in the processing. [%(default)s].')
    p.add_argument(
        '--mask', metavar='',
        help='Path to a binary mask. Only the data inside the mask will be '
             'used for computations and reconstruction [%(default)s].')
    p.add_argument(
        '--rt', dest='r_threshold', type=float, default='0.1',
        help='Relative threshold on fODF amplitude in percentage  '
             '[%(default)s].')
    p.add_argument(
        '--processes', dest='nbr_processes', metavar='NBR', type=int,
        help='Number of sub processes to start. Default : cpu count')
    add_sh_basis_args(p)
    add_overwrite_arg(p)
    p.add_argument(
        '--vis', dest='visu', action='store_true',
        help='Export map for better visualization in FiberNavigator.\n'
             '!WARNING! these maps should not be used to compute statistics  '
             '[%(default)s].')
    p.add_argument(
        '--not_all', action='store_true',
        help='If set, only saves the files specified using the file flags  '
             '[%(default)s].')

    g = p.add_argument_group(title='File flags')
    g.add_argument(
        '--afd', metavar='file', default='',
        help='Output filename for the AFD_max map.')
    g.add_argument(
        '--afd_total', metavar='file', default='',
        help='Output filename for the AFD_total map (SH coeff = 0).')
    g.add_argument(
        '--afd_sum', metavar='file', default='',
        help='Output filename for the sum of all peak contributions (sum of '
             'fODF lobes on the sphere).')
    g.add_argument(
        '--gfa', metavar='file', default='',
        help='Output filename for the GFA map.')
    g.add_argument(
        '--nufo', metavar='file', default='',
        help='Output filename for the NuFO map.')
    g.add_argument(
        '--qa', metavar='file', default='',
        help='Output filename for the QA map.')
    g.add_argument(
        '--rgb', metavar='file', default='',
        help='Output filename for the RGB map.')
    g.add_argument(
        '--peaks', metavar='file', default='',
        help='Output filename for the extracted peaks.')
    return p


def save(data, affine, output, visu=False):
    if visu:
        img = nib.Nifti1Image(np.array(data, 'uint8'),  affine)
        filename, extension1 = os.path.splitext(output)
        filename, extension2 = os.path.splitext(filename)
        nib.save(img, filename+'_fibernav' + extension2 + extension1)
    else:
        img = nib.Nifti1Image(np.array(data, 'float32'),  affine)
        nib.save(img, output)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.not_all:
        args.afd = args.afd or 'afd_max.nii.gz'
        args.afd_total = args.afd_total or 'afd_total_sh0.nii.gz'
        args.afd_sum = args.afd_sum or 'afd_sum.nii.gz'
        args.nufo = args.nufo or 'nufo.nii.gz'
        args.peaks = args.peaks or 'peaks.nii.gz'

    arglist = [args.afd, args.afd_total, args.afd_sum, args.nufo, args.peaks]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least '
                     'one file to output.')

    assert_inputs_exist(parser, [])
    assert_outputs_exist(parser, args, arglist)

    vol = nib.load(args.input)
    data = vol.get_fdata(dtype=np.float32)
    affine = vol.affine

    if args.mask is None:
        mask = None
    else:
        mask = np.asanyarray(nib.load(args.mask).dataobj).astype(np.bool)
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")

    sphere = get_sphere(args.sphere)

    # Computing peaks
    peak_dirs, peak_values, peak_indices = peaks_from_sh(data,
                                                        sphere,
                                                        mask=mask,
                                                        relative_peak_threshold=.5,
                                                        absolute_threshold=args.at,
                                                        min_separation_angle=25,
                                                        normalize_peaks=True,
                                                        sh_basis_type=args.sh_basis,
                                                        nbr_processes=args.nbr_processes)

    # Computing maps
    nufo_map, afd_map, afd_sum \
        ,rgb_map, gfa_map, qa_map = maps_from_sh(data, peak_dirs,
                                                 peak_values, peak_indices,
                                                 sphere,
                                                 nbr_processes=args.nbr_processes)

    # Save result
    if args.nufo:
        save(nufo_map, affine, args.nufo)

    if args.afd:
        save(afd_map, affine, args.afd)

    if args.afd_total:
        # this is the analytical afd total
        afd_tot = data[:, :, :, 0]
        save(afd_tot, affine, args.afd_total)

    if args.afd_sum:
        save(afd_sum, affine, args.afd_sum)

    if args.gfa:
        nib.save(nib.Nifti1Image(gfa_map, affine), args.gfa)

    if args.qa:
        nib.save(nib.Nifti1Image(qa_map, affine), args.qa)

    if args.rgb:
        nib.save(nib.Nifti1Image(rgb_map.astype('uint8'), affine), args.rgb)

    if args.peaks:
        # nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peak_dirs),
                                 # affine), args.peaks)
        nib.save(nib.Nifti1Image(peak_indices, vol.affine), args.peaks)

    if args.visu:
        if nufo_map.max() > nufo_map.min():
            nufo_map = (255 * (nufo_map - nufo_map.min()) / (nufo_map.max() -
                                                             nufo_map.min()))

        if afd_map.max() > afd_map.min():
            afd_map = (255 * (afd_map - afd_map.min()) / (afd_map.max() -
                                                          afd_map.min()))

        save(nufo_map, affine, args.nufo, True)
        save(afd_map, affine, args.afd, True)


if __name__ == "__main__":
    main()
