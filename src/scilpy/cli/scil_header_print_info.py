#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prints the raw header from the provided file or only the specified keys.
Supports trk, nii and mgz files.

By default, a simple summary view is shown. Use --all for a more complete
display.
"""

import argparse
import logging
import pprint

import nibabel as nib
from nibabel import aff2axcodes

from scilpy.io.utils import assert_inputs_exist, add_verbose_arg
from scilpy.utils.filenames import split_name_with_nii
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_file',
                   help='Input file (trk, nii and mgz).')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--raw_header', action='store_true',
                   help="Print the whole header as received by nibabel.")
    g.add_argument('--keys', nargs='+',
                   help="Print only the specified keys from nibabel's header")

    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_file)

    _, in_extension = split_name_with_nii(args.in_file)

    if in_extension in ['.tck', '.trk']:
        header = nib.streamlines.load(args.in_file, lazy_load=True).header
    elif in_extension in ['.nii', '.nii.gz', '.mgz']:
        header = dict(nib.load(args.in_file).header)
    else:
        parser.error('{} is not a supported extension.'.format(in_extension))

    if args.keys:
        for key in args.keys:
            if key not in header:
                parser.error('Key {} is not in the header of {}.'.format(key,
                             args.in_file))
            print(" '{}': {}".format(key, header[key]))
    elif args.raw_header:
        pp = pprint.PrettyPrinter(indent=1)
        pp.pprint(header)
    else:
        # mrinfo-type of print.

        # Getting info:
        if in_extension in ['.nii', '.nii.gz', '.mgz']:
            img = nib.load(args.in_file)  # Data type and affine require img
            affine = img.affine
            dtype = img.get_data_dtype().type
            if in_extension in ['.nii', '.nii.gz']:
                nb_dims = header['dim'][0]
                dims = header['dim'][1:1 + nb_dims]
                vox_size = header['dim'][5:5 + nb_dims]
            elif in_extension == '.mgz':
                dims = header['dims']
                vox_size = header['delta']
        else:
            affine = nib.streamlines.load(args.in_file, lazy_load=True).affine

            # 'dimensions' and 'voxel_sizes' exist for both tck and trk
            dims = header['dimensions']
            vox_size = header['voxel_sizes']

            if in_extension == '.trk':
                dtype = 'float32' # always float32 for trk
            else:
                dtype = header['datatype']

        print("**************************************")
        print("File name:    {}".format(args.in_file))
        print("**************************************")
        print("  Dimensions: {}".format(dims))
        print("  Voxel size: {}".format(vox_size))
        print("  Datatype: {}".format(dtype))
        print("  Orientation: {}".format(aff2axcodes(affine)))
        print("  Afine (vox2rasmm):\n{}".format(affine))


if __name__ == "__main__":
    main()
