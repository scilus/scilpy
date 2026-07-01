#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Fiber orientations analysis using Frangi filters as in [1]. Given an input grayscale
image, the script estimates the local orientation and probability of tube-like structures.
By default, the script uses a single scale, but more scales can be provided with the
--sigma argument. When more than one scale are provided, the script returns the maximum
response across all scales.

Parameters --alpha, --beta and --gamma are used to control the sensitivity of the filter to
different structures. The defaults correspond to the original Frangi filter paper [2], but
they can be adjusted to better fit the data. The argument --vesselness_threshold is used to
filter out low-probability orientations.

The memory requirement scales with the size of the input image. For example, a whole-mouse
brain S-OCT image at 10 microns (dimensions 828 x 882 x 871; 1.5 GB compressed file) requires
320 GB of RAM. The time requirement scales with the number of scales provided. For example,
the same whole-mouse brain requires 4.5 hrs to process on a single core (e.g. Rorqual) using
4 scales. On a smaller test image (dimensions 768 x 768 x 36; 151 MB compressed file), the
script requires around 6 GB of RAM and 1 min to process using 4 scales on a single core.
"""
import argparse
import logging
import nibabel as nib
import numpy as np

from scilpy.feature.orientation import frangi_filter
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist, add_overwrite_arg
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
    p.add_argument('--alpha', default=0.5, type=float,
                   help='Alpha parameter controlling sensitivity to plate-like structures. \n'
                        'The higher `alpha` the less likely we are to label flat structures as tubes. [%(default)s]')
    p.add_argument('--beta', default=0.5, type=float,
                   help='Beta parameter controlling sensitivity to locally-isotropic structures (blobs).\n'
                        'The higher `beta` the less likely we are to label blobs as tubes. [%(default)s]')
    p.add_argument('--gamma', type=float,
                   help='Correction constant that adjusts the sensitivity to areas\n'
                        'of high variance/texture/structure. By default, half of the\n'
                        'maximum Hessian norm.')
    p.add_argument('--sigma', nargs='+', type=float, default=1.0,
                   help='Sigmas used in voxel space. Can be a single value.[%(default)s]')
    p.add_argument('--vesselness_threshold', type=float, default=0.0,
                   help='Vesselness threshold for accepting a direction. [%(default)s]')
    p.add_argument('--padding_mode', default='constant',
                   choices=['constant', 'edge', 'symmetric', 'reflect', 'wrap'],
                   help='Padding mode for Frangi filter. [%(default)s]')
    p.add_argument('--padding_cval', type=float, default=0.0,
                   help='Constant value for padding. Only used if padding_mode is constant. [%(default)s]')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image)
    assert_outputs_exist(parser, args, [args.out_direction, args.out_probability])

    in_im = nib.load(args.in_image)
    in_data = in_im.get_fdata().astype(np.float32)

    scales = np.atleast_1d(args.sigma)
    prob, direction = frangi_filter(in_data, scales, alpha=args.alpha,
                                    beta=args.beta, gamma=args.gamma,
                                    threshold=args.vesselness_threshold,
                                    padding_mode=args.padding_mode,
                                    padding_cval=args.padding_cval)

    # write outputs
    nib.save(nib.Nifti1Image(direction.astype(np.float32), in_im.affine), args.out_direction)
    nib.save(nib.Nifti1Image(prob.astype(np.float32), in_im.affine), args.out_probability)


if __name__ == '__main__':
    main()
