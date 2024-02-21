#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Count the number of non-zero voxels in an image file.

If you give it an image with more than 3 dimensions, it will summarize the 4th
(or more) dimension to one voxel, and then find non-zero voxels over this.
This means that if there is at least one non-zero voxel in the 4th dimension,
this voxel of the 3D volume will be considered as non-zero.

Formerly: scil_count_non_zero_voxels.py
"""

import argparse
import logging
import os

import nibabel as nib
import numpy as np

from scilpy.image.volume_operations import count_non_zero_voxels
from scilpy.io.utils import assert_inputs_exist, add_verbose_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image', metavar='IN_FILE',
                   help='Input file name, in nifti format.')

    p.add_argument(
        '--out', metavar='OUT_FILE', dest='out_filename',
        help='name of the output file, which will be saved as a text file.')
    p.add_argument(
        '--stats', action='store_true', dest='stats_format',
        help='output the value using a stats format. Using this syntax will '
             'append\na line to the output file, instead of creating a file '
             'with only one line.\nThis is useful to create a file to be used '
             'as the source of data for a graph.\nCan be combined with --id')
    p.add_argument(
        '--id', dest='value_id',
        help='Id of the current count. If used, the value of this argument '
             'will be\noutput (followed by a ":") before the count value.\n'
             'Mostly useful with --stats.')

    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_image)
    # out_filename can exist or not

    # Load image file
    im = nib.load(args.in_image).get_fdata(dtype=np.float32)

    nb_voxels = count_non_zero_voxels(im)

    if args.out_filename is not None:
        open_mode = 'w'
        add_newline = False
        if args.stats_format and os.path.exists(args.out_filename):
            open_mode = 'a'

            # Check if file is empty or not
            if os.stat(args.out_filename).st_size > 0:
                add_newline = True

        with open(args.out_filename, open_mode) as out_file:
            if add_newline:
                out_file.write('\n')
            if args.value_id:
                out_file.write(args.value_id + ' ')
            out_file.write(str(nb_voxels))
    else:
        print(nb_voxels)


if __name__ == "__main__":
    main()
