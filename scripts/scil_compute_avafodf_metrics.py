#!/usr/bin/env python3

"""
Compute odd-power map, peaks directions and values
and nupeaks maps for ava-fodf.
"""

import argparse
import nibabel as nib
import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import order_from_ncoef, sph_harm_full_ind_list
from scilpy.io.utils import (add_overwrite_arg,
                             add_sh_basis_args,
                             save_matrix_in_any_format,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.multi_processes import peaks_from_sh


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_avafodf',
                   help='Input ava-fodf file.')
    p.add_argument('in_fodf',
                   help='FODF file used as ava-fodf input.')
    p.add_argument('in_mask',
                   help='Input WM mask.')

    p.add_argument('--avafodf_peaks_dirs', default='avafodf_peaks.nii.gz',
                   help='Output peaks directions file name.')
    p.add_argument('--avafodf_peaks_vals', default='avafodf_peaks_vals.nii.gz',
                   help='Output peaks values file name.')
    p.add_argument('--fodf_peaks_dirs', default='fodf_peaks.nii.gz',
                   help='Output peaks directions file name.')
    p.add_argument('--fodf_peaks_vals', default='fodf_peaks_vals.nii.gz',
                   help='Output peaks values file name.')
    p.add_argument('--avafodf_nupeaks', default='avafodf_nupeaks.nii.gz',
                   help='Output ava-fodf nupeaks file name.')
    p.add_argument('--fodf_nupeaks', default='fodf_nupeaks.nii.gz',
                   help='Output fodf nupeaks file name.')
    p.add_argument('--nupeaks_compare', default='nupeaks_compare.npy',
                   help='Output nupeaks compare file name.')
    p.add_argument('--odd_pwr_map', default='avafodf_odd_pwr_map.nii.gz',
                   help='Output odd power map file name.')
    p.add_argument('--avafodf_crossings', default='avafodf_crossings.txt',
                   help='Output avafodf crossings proportion file name.')
    p.add_argument('--fodf_crossings', default='fodf_crossings.txt',
                   help='Output fodf crossings proportion file name.')

    p.add_argument('--sphere', default='repulsion724',
                   help='Sphere to use for SH projection.')
    p.add_argument('--rel_peaks_threshold', default=0.3, type=float,
                   help='Relative threshold for peak extraction.')
    p.add_argument('--abs_peaks_threshold', default=0.1, type=float,
                   help='Absolute peaks threshold for peaks extraction.')
    p.add_argument('--min_sep_angle', default=25, type=float,
                   help='Minimum separation angle for peaks extraction.')
    p.add_argument('--npeaks', default=10, type=int,
                   help='Maximum number of peaks to extract.')

    add_overwrite_arg(p)
    add_sh_basis_args(p)
    return p


def compute_peaks(data, mask, sphere, sh_basis, full_basis,
                  rel_th, abs_th, min_angle, npeaks):
    peak_dirs, peak_values, \
        peak_indices = peaks_from_sh(data, sphere, mask=mask,
                                     relative_peak_threshold=rel_th,
                                     absolute_threshold=abs_th,
                                     min_separation_angle=min_angle,
                                     npeaks=npeaks,
                                     sh_basis_type=sh_basis,
                                     full_basis=full_basis,
                                     is_symmetric=False)
    return peak_dirs, peak_values


def compute_nupeaks(peaks_values):
    nonzero_peaks = peaks_values > 0.
    nupeaks = np.cumsum(nonzero_peaks, axis=-1)[..., -1]

    return nupeaks.astype(np.uint8)


def compare_nupeaks_in_wm(sym_nupeaks, asym_nupeaks, wm_mask):
    wm_sym_nupeaks = sym_nupeaks[wm_mask]
    wm_asym_nupeaks = asym_nupeaks[wm_mask]

    max_npeaks = max(wm_sym_nupeaks.max(), wm_asym_nupeaks.max())
    nupeaks_compare = np.zeros((max_npeaks + 1, max_npeaks + 1))

    # Elements along first axis represent symmetric nupeaks while
    # elements along last axis represent asymmetric nupeaks
    for npeaks in range(max_npeaks + 1):
        npeaks_mask = wm_sym_nupeaks == npeaks
        masked_asym_nupeaks = wm_asym_nupeaks[npeaks_mask]
        # Number of occurence of each number of peaks
        bincount = np.bincount(masked_asym_nupeaks)
        nupeaks_compare[npeaks, :bincount.size] =\
            bincount / masked_asym_nupeaks.size

    return nupeaks_compare


def compute_odd_power_map(data):
    order = order_from_ncoef(data.shape[-1], is_full_basis=True)
    _, l_list = sph_harm_full_ind_list(order)
    odd_order_coeffs = data[..., l_list % 2 == 1]
    odd_order_norms = np.linalg.norm(odd_order_coeffs, axis=-1)
    full_norms = np.linalg.norm(data, axis=-1)

    odd_pwr_map = np.zeros_like(full_norms)
    mask = full_norms > 0

    odd_pwr_map[mask] = odd_order_norms[mask] / full_norms[mask]

    return odd_pwr_map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Safety checks for inputs
    inputs = [args.in_avafodf, args.in_fodf, args.in_mask]
    assert_inputs_exist(parser, inputs)

    # Safety checks for outputs
    outputs = [args.avafodf_peaks_dirs,
               args.avafodf_peaks_vals,
               args.avafodf_nupeaks,
               args.fodf_peaks_dirs,
               args.fodf_peaks_vals,
               args.fodf_nupeaks,
               args.nupeaks_compare]
    assert_outputs_exist(parser, args, outputs)

    # Load inputs
    avafodf_img = nib.load(args.in_avafodf)
    avafodf_data = avafodf_img.get_fdata(dtype=np.float)

    fodf_img = nib.load(args.in_fodf)
    fodf_data = fodf_img.get_fdata(dtype=np.float)

    mask = get_data_as_mask(nib.load(args.in_mask), dtype=np.bool)

    sphere = get_sphere(args.sphere)

    # Compute ava-fodf peaks
    peak_dirs, peak_vals = compute_peaks(avafodf_data, mask, sphere,
                                         args.sh_basis, True,
                                         args.rel_peaks_threshold,
                                         args.abs_peaks_threshold,
                                         args.min_sep_angle,
                                         args.npeaks)
    nib.save(nib.Nifti1Image(peak_dirs, avafodf_img.affine),
             args.avafodf_peaks_dirs)
    nib.save(nib.Nifti1Image(peak_vals, avafodf_img.affine),
             args.avafodf_peaks_vals)

    # Compute ava-fodf nupeaks
    avafodf_nupeaks = compute_nupeaks(peak_vals)
    nib.save(nib.Nifti1Image(avafodf_nupeaks, avafodf_img.affine),
             args.avafodf_nupeaks)

    # Compute fodf peaks
    peak_dirs, peak_vals = compute_peaks(fodf_data, mask, sphere,
                                         args.sh_basis, False,
                                         args.rel_peaks_threshold,
                                         args.abs_peaks_threshold,
                                         args.min_sep_angle,
                                         args.npeaks)
    nib.save(nib.Nifti1Image(peak_dirs, fodf_img.affine),
             args.fodf_peaks_dirs)
    nib.save(nib.Nifti1Image(peak_vals, fodf_img.affine),
             args.fodf_peaks_vals)

    # Compute fodf nupeaks
    fodf_nupeaks = compute_nupeaks(peak_vals)
    nib.save(nib.Nifti1Image(fodf_nupeaks, fodf_img.affine),
             args.fodf_nupeaks)

    # Compare nupeaks
    save_matrix_in_any_format(
        args.nupeaks_compare,
        compare_nupeaks_in_wm(fodf_nupeaks, avafodf_nupeaks, mask)
    )

    # Compute odd-power map for ava-fodf
    odd_pwr_map = compute_odd_power_map(avafodf_data)
    nib.save(nib.Nifti1Image(odd_pwr_map.astype(np.float32),
                             avafodf_img.affine), args.odd_pwr_map)

    # Compute proportion of crossings before/after filtering
    fodf_crossings_proportions =\
        np.count_nonzero(fodf_nupeaks[mask] > 2) / np.count_nonzero(mask)
    save_matrix_in_any_format(args.avafodf_crossings,
                              np.array([fodf_crossings_proportions]))

    avafodf_crossings_proportions =\
        np.count_nonzero(avafodf_nupeaks[mask] > 2) / np.count_nonzero(mask)
    save_matrix_in_any_format(args.fodf_crossings,
                              np.array([avafodf_crossings_proportions]))


if __name__ == '__main__':
    main()
