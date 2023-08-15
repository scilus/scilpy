#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper for SyNb0 available in Dipy, to run it on a single subject.
Requires Skull-Strip b0 and t1w images as input, the script will normalize the
t1w's WM to 110, co-register both images, then register it to the appropriate
template, run SyNb0 and then transform the result back to the original space.

This script must be used carefully, as it is not meant to be used in an
environment with the following dependencies already installed (not default
in Scilpy):
- tensorflow-addons
- tensorrt
- tensorflow
"""


import argparse
import logging
import os
import sys
import warnings

# Disable tensorflow warnings
with warnings.catch_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.simplefilter("ignore")
    from dipy.nn.synb0 import Synb0

from dipy.align.imaffine import AffineMap
from dipy.segment.tissue import TissueClassifierHMRF
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.image import get_synb0_template_path, register_image


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_b0',
                   help='Input skull-stripped b0 image.')
    p.add_argument('in_t1',
                   help='Input skull-stripped t1w image.')
    p.add_argument('out_b0',
                   help='Output skull-stripped b0 image without distortion.')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, [args.in_b0, args.in_t1])
    assert_outputs_exist(parser, args, args.out_b0)

    if not args.verbose:
        sys.stdout = open(os.devnull, 'w')
    else:
        logging.getLogger().setLevel(logging.INFO)

    template_img = nib.load(get_synb0_template_path())
    template_data = template_img.get_fdata()

    b0_img = nib.load(args.in_b0)
    b0_data = b0_img.get_fdata()
    t1_img = nib.load(args.in_t1)
    t1_data = gaussian_filter(t1_img.get_fdata(), sigma=1)

    # Crude estimation of the WM mean intensity and normalization
    logging.info('Estimating WM mean intensity')
    hmrf = TissueClassifierHMRF()
    _, final_segmentation, _ = hmrf.classify(t1_data, 3, 0.25,
                                             tolerance=1e-4, max_iter=2)
    avg_wm = np.mean(t1_data[final_segmentation == 3])
    t1_data /= avg_wm
    t1_data *= 110

    # SyNB0 works only in a standard space, so we need to register the images
    logging.info('Registering images')
    t1_to_b0, _ = register_image(b0_data, b0_img.affine,
                                 t1_data, t1_img.affine)
    t1_to_b0_to_mni, t1_to_b0_to_mni_transform = register_image(template_data,
                                                                template_img.affine,
                                                                t1_to_b0,
                                                                b0_img.affine)
    affine_map = AffineMap(t1_to_b0_to_mni_transform,
                           template_data.shape, template_img.affine,
                           b0_data.shape, b0_img.affine)
    b0_to_mni = affine_map.transform(b0_data.astype(np.float64))

    logging.info('Running SyN-B0')
    SyNb0 = Synb0(args.verbose)
    rev_b0 = SyNb0.predict(b0_to_mni, t1_to_b0_to_mni)
    rev_b0 = affine_map.transform_inverse(rev_b0.astype(np.float64))

    dtype = b0_img.get_data_dtype()
    nib.save(nib.Nifti1Image(rev_b0.astype(dtype), b0_img.affine),
             args.out_b0)


if __name__ == "__main__":
    main()
