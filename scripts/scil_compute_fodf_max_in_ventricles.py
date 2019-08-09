#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute the maximum fODF in the ventricles.
"""

from __future__ import division, print_function
from builtins import str
from builtins import range
from past.utils import old_div
import argparse
import logging

from dipy.data import get_sphere
import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_sh_basis_args,
                             assert_inputs_exist, assert_outputs_exists)
from scilpy.reconst.utils import find_order_from_nb_coeff, get_b_matrix


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input',  metavar='fODFs',
                   help='Path of the fODF volume in spherical harmonics (SH).')
    p.add_argument('fa',  metavar='FA',
                   help='Path to the FA volume.')
    p.add_argument('md',  metavar='MD',
                   help='Path to the mean diffusivity (MD) volume.')

    p.add_argument(
        '--fa_t', dest='fa_threshold',  type=float, default='0.1',
        help='Maximal threshold of FA (voxels under that threshold are '
             'considered for evaluation, default: 0.1)')
    p.add_argument(
        '--md_t', dest='md_threshold',  type=float, default='0.003',
        help='Minimal threshold of MD in mm2/s (voxels above that threshold '
             'are considered for evaluation, default: 0.003)')
    p.add_argument(
        '--max_value_output',  metavar='file',
        help='Output path for the text file containing the value. If not set '
             'the file will not be saved.')
    p.add_argument(
        '--mask_output',  metavar='file',
        help='Output path for the ventricule mask. If not set, the mask will '
             'not be saved.')
    add_sh_basis_args(p)
    add_overwrite_arg(p)
    p.add_argument('-v', action='store_true', dest='verbose',
                   help='Use verbose output. Default: false.')
    return p


def load(path):
    img = nib.load(path)
    return img.get_data(), img.affine, img.header.get_zooms()[:3]


def save(data, affine, output):
    img = nib.Nifti1Image(np.array(data, 'float32'),  affine)
    nib.save(img, output)


def get_ventricles_max_fodf(data, fa, md, zoom, args):
    order = find_order_from_nb_coeff(data)
    sphere = get_sphere('repulsion100')
    b_matrix = get_b_matrix(order, sphere, args.sh_basis)
    sum_of_max = 0
    count = 0

    mask = np.zeros(data.shape[:-1])

    if np.min(data.shape[:-1]) > 40:
        step = 20
    else:
        if np.min(data.shape[:-1]) > 20:
            step = 10
        else:
            step = 5

    # 1000 works well at 2x2x2 = 8 mm^3
    # Hence, we multiply by the volume of a voxel
    vol = (zoom[0] * zoom[1] * zoom[2])
    if vol != 0:
        max_number_of_voxels = old_div(1000 * 8, vol)
    else:
        max_number_of_voxels = 1000

    all_i = list(range(int(data.shape[0]/2) - step, int(data.shape[0]/2) + step))
    all_j = list(range(int(data.shape[1]/2) - step, int(data.shape[1]/2) + step))
    all_k = list(range(int(data.shape[2]/2) - step, int(data.shape[2]/2) + step))
    for i in all_i:
        for j in all_j:
            for k in all_k:
                if count > max_number_of_voxels - 1:
                    continue
                if fa[i, j, k] < args.fa_threshold \
                        and md[i, j, k] > args.md_threshold:
                    sf = np.dot(data[i, j, k], b_matrix.T)
                    sum_of_max += sf.max()
                    count += 1
                    mask[i, j, k] = 1

    logging.debug('Number of voxels detected: %s', count)
    if count == 0:
        logging.warning('No voxels found for evaluation! Change your fa '
                        'and/or md thresholds')
        return 0, mask

    logging.debug('Average max fodf value: %s', sum_of_max / count)
    return sum_of_max / count, mask


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.input, args.fa, args.md])
    assert_outputs_exist(parser, args, [],
                          [args.max_value_output, args.mask_output])

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load input image
    fodf, affine, zoom = load(args.input)

    fa, _, _ = load(args.fa)
    md, _, _ = load(args.md)

    value, mask = get_ventricles_max_fodf(fodf, fa, md, zoom, args)

    if args.mask_output:
        save(mask, affine, args.mask_output)

    if args.max_value_output:
        text_file = open(args.max_value_output, "w")
        text_file.write(str(value))
        text_file.close()
    else:
        print("Maximal value in ventricles: {}".format(value))


if __name__ == "__main__":
    main()
