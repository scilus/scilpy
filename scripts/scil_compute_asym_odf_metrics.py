#!/usr/bin/env python3
"""
Script to compute various metrics derivated from asymmetric ODF.

These metrics include an asymmetric peak directions image, a number of peaks
(nupeaks) map [2], a cosine-similarity-based asymmetry map [1] and an
odd-power map [2].

The asymmetric peak directions image contains peaks per hemisphere, considering
antipodal sphere directions as distinct. On a symmetric signal, the number of
asymmetric peaks extracted is then twice the number of symmetric peaks.

The nupeaks map is the asymmetric alternative to NuFO maps. It counts the
number of asymmetric peaks extracted and ranges in [0..N] with N the maximum
number of peaks.

The cosine-based asymmetry map is in the range [0..1], with 0 corresponding
to a perfectly symmetric signal and 1 to a perfectly asymmetric signal.

The odd-power map is also in the range [0..1], with 0 corresponding
to a perfectly symmetric signal and 1 to a perfectly anti-symmetric signal. It
is given by the ratio of the L2-norm of odd SH coefficients on the L2-norm of
all SH coefficients.
"""


import argparse
import nibabel as nib
import numpy as np

from dipy.data import get_sphere, SPHERE_FILES
from dipy.direction.peaks import reshape_peaks_for_visualization
from dipy.reconst.shm import sph_harm_ind_list

from scilpy.reconst.multi_processes import peaks_from_sh
from scilpy.reconst.utils import get_sh_order_and_fullness
from scilpy.io.utils import (add_processes_arg,
                             add_sh_basis_args,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.io.image import get_data_as_mask


EPILOG = """
References:
[1] S. Cetin Karayumak, E. Özarslan, and G. Unal,
“Asymmetric Orientation Distribution Functions (AODFs) revealing
intravoxel geometry in diffusion MRI,” Magnetic Resonance Imaging,
vol. 49, pp. 145–158, Jun. 2018, doi: 10.1016/j.mri.2018.03.006.

[2] C. Poirier, E. St-Onge, and M. Descoteaux, "Investigating the Occurence of
Asymmetric Patterns in White Matter Fiber Orientation Distribution Functions"
[Abstract], In: Proc. Intl. Soc. Mag. Reson. Med. 29 (2021), 2021 May 15-20,
Vancouver, BC, Abstract number 0865.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_sh', help='Input SH image.')

    p.add_argument('--mask', default='',
                   help='Optional mask.')

    # outputs
    p.add_argument('--cos_asym_map', default='',
                   help='Output asymmetry map using cos similarity.')
    p.add_argument('--odd_power_map', default='',
                   help='Output odd power map.')
    p.add_argument('--peaks', default='',
                   help='Output filename for the extracted peaks.')
    p.add_argument('--peak_values', default='',
                   help='Output filename for the extracted peaks values.')
    p.add_argument('--peak_indices', default='',
                   help='Output filename for the generated peaks indices on '
                        'the sphere.')
    p.add_argument('--nupeaks', default='',
                   help='Output filename for the nupeaks file.')
    p.add_argument('--not_all', action='store_true',
                   help='If set, only saves the files specified using the '
                        'file flags [%(default)s].')

    p.add_argument('--at', dest='a_threshold', type=float, default='0.0',
                   help='Absolute threshold on fODF amplitude. This '
                        'value should be set to\napproximately 1.5 to 2 times '
                        'the maximum fODF amplitude in isotropic voxels\n'
                        '(ie. ventricles).\n'
                        'Use compute_fodf_max_in_ventricles.py to find the '
                        'maximal value.\n'
                        'See [Dell\'Acqua et al HBM 2013] [%(default)s].')
    p.add_argument('--rt', dest='r_threshold', type=float, default='0.1',
                   help='Relative threshold on fODF amplitude in percentage '
                        '[%(default)s].')
    p.add_argument('--sphere', default='symmetric724',
                   choices=sorted(SPHERE_FILES.keys()),
                   help='Sphere to use for peak directions estimation '
                        '[%(default)s].')

    add_processes_arg(p)
    add_sh_basis_args(p)
    add_overwrite_arg(p)
    return p


def compute_cos_asym_map(sh_coeffs, order, mask):
    _, l_list = sph_harm_ind_list(order, full_basis=True)

    sign = np.power(-1.0, l_list)
    sign = np.reshape(sign, (1, 1, 1, len(l_list)))
    sh_squared = sh_coeffs**2
    mask = np.logical_and(sh_squared.sum(axis=-1) > 0., mask)

    cos_asym_map = np.zeros(sh_coeffs.shape[:-1])
    cos_asym_map[mask] = np.sum(sh_squared * sign, axis=-1)[mask] / \
        np.sum(sh_squared, axis=-1)[mask]

    cos_asym_map = np.sqrt(1 - cos_asym_map**2) * mask

    return cos_asym_map


def compute_odd_power_map(sh_coeffs, order, mask):
    _, l_list = sph_harm_ind_list(order, full_basis=True)
    odd_l_list = (l_list % 2 == 1).reshape((1, 1, 1, -1))

    odd_order_norm = np.linalg.norm(sh_coeffs * odd_l_list,
                                    ord=2, axis=-1)

    full_order_norm = np.linalg.norm(sh_coeffs, ord=2, axis=-1)

    asym_map = np.zeros(sh_coeffs.shape[:-1])
    mask = np.logical_and(full_order_norm > 0, mask)
    asym_map[mask] = odd_order_norm[mask] / full_order_norm[mask]

    return asym_map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.not_all:
        args.cos_asym_map = args.cos_asym_map or 'cos_asym_map.nii.gz'
        args.odd_power_map = args.odd_power_map or 'odd_power_map.nii.gz'
        args.peaks = args.peaks or 'asym_peaks.nii.gz'
        args.peak_values = args.peak_values or 'asym_peak_values.nii.gz'
        args.peak_indices = args.peak_indices or 'asym_peak_indices.nii.gz'
        args.nupeaks = args.nupeaks or 'nupeaks.nii.gz'

    arglist = [args.cos_asym_map, args.odd_power_map, args.peaks,
               args.peak_values, args.peak_indices, args.nupeaks]
    if args.not_all and not any(arglist):
        parser.error('When using --not_all, you need to specify at least '
                     'one file to output.')

    inputs = [args.in_sh]
    if args.mask:
        inputs.append(args.mask)

    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, arglist)

    sh_img = nib.load(args.in_sh)
    sh = sh_img.get_fdata()

    sphere = get_sphere(args.sphere)

    sh_order, full_basis = get_sh_order_and_fullness(sh.shape[-1])
    if not full_basis:
        parser.error('Invalid SH image. A full SH basis is expected.')

    if args.mask:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
    else:
        mask = np.sum(np.abs(sh), axis=-1) > 0

    if args.cos_asym_map:
        cos_asym_map = compute_cos_asym_map(sh, sh_order, mask)
        nib.save(nib.Nifti1Image(cos_asym_map, sh_img.affine),
                 args.cos_asym_map)

    if args.odd_power_map:
        odd_power_map = compute_odd_power_map(sh, sh_order, mask)
        nib.save(nib.Nifti1Image(odd_power_map, sh_img.affine),
                 args.odd_power_map)

    if args.peaks or args.peak_values or args.peak_indices or args.nupeaks:
        peaks, values, indices =\
            peaks_from_sh(sh, sphere, mask=mask,
                          relative_peak_threshold=args.r_threshold,
                          absolute_threshold=args.a_threshold,
                          min_separation_angle=25,
                          normalize_peaks=False,
                          # because v and -v are unique, we want twice
                          # the usual default value (5) of npeaks
                          npeaks=10,
                          sh_basis_type=args.sh_basis,
                          nbr_processes=args.nbr_processes,
                          full_basis=True,
                          is_symmetric=False)

        if args.peaks:
            nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks),
                                     sh_img.affine), args.peaks)

        if args.peak_values:
            nib.save(nib.Nifti1Image(values, sh_img.affine),
                     args.peak_values)

        if args.peak_indices:
            nib.save(nib.Nifti1Image(indices.astype(np.uint8), sh_img.affine),
                     args.peak_indices)

        if args.nupeaks:
            nupeaks = np.count_nonzero(values, axis=-1).astype(np.uint8)
            nib.save(nib.Nifti1Image(nupeaks, sh_img.affine), args.nupeaks)


if __name__ == '__main__':
    main()
