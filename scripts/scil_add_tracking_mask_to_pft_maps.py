#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modify PFT maps to allow PFT tracking in given mask (e.g edema).
"""

import argparse

import nibabel as nib

from scilpy.io.utils import (
    add_overwrite_arg, assert_inputs_exist, assert_outputs_exist)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('map_include',
                        help='PFT map include.')
    parser.add_argument('map_exclude',
                        help='PFT map exclude.')
    parser.add_argument('additional_mask',
                        help='Allow PFT tracking in this mask.')
    parser.add_argument('map_include_corr',
                        help='Corrected PFT map include output file name.')
    parser.add_argument('map_exclude_corr',
                        help='Corrected PFT map exclude output file name.')
    add_overwrite_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.map_include, args.map_exclude,
                                 args.additional_mask])
    assert_outputs_exist(parser, args, [args.map_include_corr,
                                        args.map_exclude_corr])

    map_inc = nib.load(args.map_include)
    map_inc_data = map_inc.get_data()

    map_exc = nib.load(args.map_exclude)
    map_exc_data = map_exc.get_data()

    additional_mask = nib.load(args.additional_mask)
    additional_mask_data = additional_mask.get_data()

    map_inc_data[additional_mask_data > 0] = 0
    map_exc_data[additional_mask_data > 0] = 0

    nib.save(
        nib.Nifti1Image(map_inc_data.astype('float32'),
                        map_inc.affine,
                        map_inc.header),
        args.map_include_corr)
    nib.save(
        nib.Nifti1Image(map_exc_data.astype('float32'),
                        map_exc.affine,
                        map_exc.header),
        args.map_exclude_corr)


if __name__ == '__main__':
    main()
