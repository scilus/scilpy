#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Fiber orientations analysis using either Frangi filters or structure tensor analysis.

By default, the script uses Frangi filters as in [1]. Given an input grayscale image,
the script estimates the local orientation and probability of tube-like structures. By
default, the script uses a single scale, but more scales can be provided with the --sigma
argument. When more than one scale are provided, the script returns the maximum response
across all scales.

Parameters --alpha, --beta and --gamma are used to control the sensitivity of the filter to
different structures. The defaults correspond to the original Frangi filter paper [2], but
they can be adjusted to better fit the data. The argument --prob_threshold is used to
filter out low-probability orientations.

The script also provides the single-scale structure tensor method as an alternative. Because
structure tensor is based on 1st-order derivatives, it is generally less sensitive to noise than
the Frangi filter, which uses 2nd-order derivatives. Also note that the definition for the
probability map differs between both methods. For structure tensor analysis, it is a simple
ratio of the 3 eigenvalues, while for Frangi filters, it is a more complex function of the 
eigenvalues relying on alpha, beta and gamma parameters [1].

The memory requirement scales with the size of the input image. For example, a whole-mouse
brain S-OCT image at 10 microns (dimensions 828 x 882 x 871; 1.5 GB compressed file) requires
320 GB of RAM. The time requirement scales with the number of scales provided. For example,
the same whole-mouse brain requires 4.5 hrs to process on a single core (e.g. Rorqual) using
4 scales. On a smaller test image (dimensions 768 x 768 x 36; 151 MB compressed file), the
script requires around 6 GB of RAM and 1 min to process using 4 scales on a single core.
"""
import argparse
import logging
from networkx import sigma
import nibabel as nib
import numpy as np

from scilpy.feature.orientation import frangi_filter
from skimage.feature import structure_tensor
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist, add_overwrite_arg, add_verbose_arg
from scilpy.version import version_string

EPILOG="""
[1] Sorelli et al, 2023, "Fiber enhancement and 3D orientation analysis in label-free
    two-photon fluorescence microscopy", Scientific Reports (2023) 13:4160
[2] Frangi et al, 1998, "Multiscale vessel enhancement filtering", Medical Image Computing
    and Computer-Assisted Intervention (MICCAI), 130-137
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG+version_string,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image', help='Input nifti image.')
    p.add_argument('out_direction', help='Output direction nifti image.')
    p.add_argument('out_probability', help='Output probability nifti image.')
    p.add_argument('--method', default='frangi', choices=['frangi', 'structure_tensor'],
                   help='Method to use for local orientation estimation. [%(default)s]')
    p.add_argument('--sigma', nargs='+', type=float, default=1.0,
                   help='Sigmas used, in voxel space. For Frangi method, this can'
                        'be a list. [%(default)s]')
    p.add_argument('--prob_threshold', type=float, default=0.0,
                   help='Probability threshold for accepting a direction. [%(default)s]')

    frangi_group = p.add_argument_group('Frangi filter parameters')
    frangi_group.add_argument('--alpha', default=0.5, type=float,
                   help='Alpha parameter controlling sensitivity to plate-like structures. \n'
                        'The higher `alpha` the less likely we are to label flat structures as tubes. [%(default)s]')
    frangi_group.add_argument('--beta', default=0.5, type=float,
                   help='Beta parameter controlling sensitivity to locally-isotropic structures (blobs).\n'
                        'The higher `beta` the less likely we are to label blobs as tubes. [%(default)s]')
    frangi_group.add_argument('--gamma', type=float,
                              help='Correction constant that adjusts the sensitivity to areas\n'
                                   'of high variance/texture/structure. By default, half of the\n'
                                   'maximum Hessian norm.')

    p.add_argument('--padding_mode', default='constant',
                   choices=['constant', 'edge', 'symmetric', 'reflect', 'wrap'],
                   help='Padding mode for Frangi filter. [%(default)s]')
    p.add_argument('--padding_cval', type=float, default=0.0,
                   help='Constant value for padding. Only used if padding_mode is constant. [%(default)s]')
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def structure_tensor_wrapper(data, sigma, padding_mode='constant', padding_cval=0):
    a_elems = structure_tensor(data, sigma=sigma, mode=padding_mode, cval=padding_cval)
    A = np.zeros(data.shape + (3, 3), dtype=np.float64)
    A[..., 0, 0] = a_elems[0]
    A[..., 0, 1] = a_elems[1]
    A[..., 0, 2] = a_elems[2]
    A[..., 1, 1] = a_elems[3]
    A[..., 1, 2] = a_elems[4]
    A[..., 2, 2] = a_elems[5]
    A[..., 1, 0] = A[..., 0, 1]
    A[..., 2, 0] = A[..., 0, 2]
    A[..., 2, 1] = A[..., 1, 2]

    return A


def divide_nonzero(num, div):
    res = np.zeros_like(num)
    nonzero_mask = div != 0
    res[nonzero_mask] = num[nonzero_mask] / div[nonzero_mask]
    return res


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image)
    assert_outputs_exist(parser, args, [args.out_direction, args.out_probability])

    if len(args.sigma) > 1 and args.method == 'structure_tensor':
        parser.error('Structure tensor method only supports a single scale. Please provide a single value for --sigma.')

    in_im = nib.load(args.in_image)
    in_data = in_im.get_fdata().astype(np.float32)

    scales = np.atleast_1d(args.sigma)
    if args.method == 'frangi':
        prob, direction = frangi_filter(in_data, scales, alpha=args.alpha,
                                        beta=args.beta, gamma=args.gamma,
                                        threshold=args.prob_threshold,
                                        padding_mode=args.padding_mode,
                                        padding_cval=args.padding_cval)

    elif args.method == 'structure_tensor':
        # Add a probability threshold
        prob = np.full(in_data.shape, args.prob_threshold)

        A = structure_tensor_wrapper(in_data, scales[0], args.padding_mode, args.padding_cval)
        evals, evecs = np.linalg.eigh(A)

        # ascending order of eigenvalues, so lambda_1 is the largest
        lambda_1 = evals[..., 2]
        lambda_2 = evals[..., 1]
        lambda_3 = evals[..., 0]

        _prob = divide_nonzero(lambda_2 - lambda_3, lambda_1)
        update_mask = _prob > prob

        prob[update_mask] = _prob[update_mask]
        direction = evecs[..., 0]
        direction[~update_mask] = 0  # set direction to zero where probability is below threshold

    # write outputs
    nib.save(nib.Nifti1Image(direction.astype(np.float32), in_im.affine), args.out_direction)
    nib.save(nib.Nifti1Image(prob.astype(np.float32), in_im.affine), args.out_probability)


if __name__ == '__main__':
    main()
