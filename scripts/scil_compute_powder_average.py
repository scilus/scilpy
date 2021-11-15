#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute powder average (mean diffusion weighted image) from set of
diffusion images.

By default will output an average image calculated from all images with
non-zero bvalue.

specify --bvalue to output an image for a single bvalue
"""

from os.path import splitext
import re

import argparse
import logging

import nibabel as nib
import numpy as np

# Aliased to avoid clashes with images called mode.
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.filenames import add_filename_suffix, split_name_with_nii
from nibabel.tmpdirs import InTemporaryDirectory


logger = logging.getLogger("Compute_Powder_Average")
logger.setLevel(logging.INFO)


# function to read bvalues from file, avoiding using dipy io which requires 
# bvec file name supplied (this is a modified version of that dipy function)
def read_bvals(fbval):
    vals = []
    if fbval is None or not fbval:
        vals.append(None)

    if not isinstance(fbval, str):
        raise ValueError('String with full path to file is required')

    base, ext = splitext(fbval)
    if ext in ['.bvals', '.bval', '.txt', '']:
        with open(fbval, 'r') as f:
            content = f.read()
            
        # We replace coma and tab delimiter by space
        with InTemporaryDirectory():
            tmp_fname = "tmp_bvals_bvecs.txt"
            with open(tmp_fname, 'w') as f:
                f.write(re.sub(r'(\t|,)', ' ', content))
            vals.append(np.squeeze(np.loadtxt(tmp_fname)))
    elif ext == '.npy':
        vals.append(np.squeeze(np.load(fbval)))
    else:
        e_s = "File type %s is not recognized" % ext
        raise ValueError(e_s)
        
    bvals = vals[0]

    if bvals is None:
        return bvals

    if len(bvals.shape) > 1:
        raise IOError('bval file should have one row')
        
    return bvals

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi',
                   help='Path of the input diffusion volume.')
    p.add_argument('in_bval',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('out_avg',
                   help='Path of the output file')
    
    add_overwrite_arg(p)
    p.add_argument('--mask',dest='mask', metavar='file',
        help='Path to a binary mask.\nOnly data inside the mask will be used '
             'for powder avg. (Default: %(default)s)')
    p.add_argument('--shell',dest='shell', metavar='int', default='',
        help='bvalue (shell) to include in powder average.\nIf not specified'
             'will include all volumes with a non-zero bvalue')
    
    p.add_argument('--shell_thr',dest='shell_thr', metavar='int', default='50',
        help='Include volumes with bvalue +- the specified threshold.\n'
             'default: 50')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval])
    
    assert_outputs_exist(parser, args, args.out_avg)

    img = nib.load(args.in_dwi)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    if args.mask is None:
        mask = None
    else:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)

    # Read bvals
    logging.info('Performing powder average')
    bvals = read_bvals(args.in_bval)

    # Select diffusion volumes to average
    if not(args.shell):
        # If no shell given, average all diffusion weigthed images
        bval_idx = bvals > 0
    else:
        min_bval = args.shell - args.shell_idx
        max_bval = args.shell + args.shell_idx
        bval_idx = np.logical_and(bvals > min_bval, bvals < max_bval)

    powder_avg = np.squeeze(np.mean(data[:,:,:,bval_idx],axis=3))
    
    powder_avg_img = nib.Nifti1Image(powder_avg.astype(np.float32), affine)
    nib.save(powder_avg_img, args.out_avg)

    del powder_avg_img

if __name__ == "__main__":
    main()
