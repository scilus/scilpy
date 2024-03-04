#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to combine labels from multiple volumes. If there is overlap, it will
overwrite them based on the input order.

    >>> scil_labels_combine.py out_labels.nii.gz
            --volume_ids animal_labels.nii 20
            --volume_ids DKT_labels.nii.gz 44 53
            --out_labels_indices 20 44 53
    >>> scil_labels_combine.py slf_labels.nii.gz
            --volume_ids a2009s_aseg.nii.gz all
            --volume_ids clean/s1__DKT.nii.gz 1028 2028

Formerly: scil_combine_labels.py.
"""


import argparse
import logging

from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels, combine_labels
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             add_verbose_arg, assert_outputs_exist)


EPILOG = """
    References:
        [1] Al-Sharif N.B., St-Onge E., Vogel J.W., Theaud G.,
            Evans A.C. and Descoteaux M. OHBM 2019.
            Surface integration for connectome analysis in age prediction.
    """


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('output',
                   help='Combined labels volume output.')

    p.add_argument('--volume_ids', nargs='+', default=[],
                   action='append', required=True,
                   help='List of volumes directly followed by their labels:\n'
                        '  --volume_ids atlasA  id1a id2a \n'
                        '  --volume_ids atlasB  id1b id2b ... \n'
                        '  "all" can be used instead of id numbers.')

    o_ids = p.add_mutually_exclusive_group()
    o_ids.add_argument('--out_labels_ids', type=int, nargs='+', default=[],
                       help='List of labels indices for output images.')
    o_ids.add_argument('--unique', action='store_true',
                       help='If set, output id with unique labels, excluding '
                            'first background value.')
    o_ids.add_argument('--group_in_m', action='store_true',
                       help='Add (x * 10 000) to each volume labels,'
                            ' where x is the input volume order number.')

    p.add_argument('--background', type=int, default=0,
                   help='Background id, excluded from output [%(default)s],\n'
                        ' the value is used as output background value.')
    p.add_argument('--merge_groups', action='store_true',
                   help='Each group from the --volume_ids option will be '
                        'merged as a single labels.')
    
    add_verbose_arg(p)
    add_overwrite_arg(p)
    
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    image_files = []
    indices_per_input_volume = []
    # Separate argument per volume
    used_indices_all = False
    for v_args in args.volume_ids:
        if len(v_args) < 2:
            parser.error("No indices was given for a given volume.")

        image_files.append(v_args[0])
        if "all" in v_args:
            used_indices_all = True
            indices_per_input_volume.append("all")
        else:
            indices_per_input_volume.append(np.asarray(v_args[1:], dtype=int))

    if used_indices_all and args.out_labels_ids:
        parser.error("'all' indices cannot be used with --out_labels_ids.")

    if args.merge_groups and (args.group_in_m or args.unique):
        parser.error("Cannot use --unique and --group_in_m with "
                     "--merge_groups.")

    # Check inputs / output
    assert_inputs_exist(parser, image_files)
    assert_outputs_exist(parser, args, args.output)

    # Load volume and do checks
    data_list = []
    first_img = nib.load(image_files[0])
    for i in range(len(image_files)):
        # Load images
        volume_nib = nib.load(image_files[i])
        data = get_data_as_labels(volume_nib)
        data_list.append(data)
        assert (is_header_compatible(first_img, image_files[i]))

        if (isinstance(indices_per_input_volume[i], str)
                and indices_per_input_volume[i] == "all"):
            indices_per_input_volume[i] = np.unique(data)

    if args.out_labels_ids:
        out_choice = ('out_labels_ids', args.out_labels_ids)
    elif args.unique:
        out_choice = ('unique',)
    elif args.group_in_m:
        out_choice = ('group_in_m',)
    else:
        out_choice = ('all_labels',)

    # Combine labels
    resulting_labels = combine_labels(data_list,
                                      indices_per_input_volume,
                                      out_choice,
                                      background_id=args.background,
                                      merge_groups=args.merge_groups)

    # Save final combined volume
    nib.save(nib.Nifti1Image(resulting_labels, first_img.affine,
                             header=first_img.header),
             args.output)


if __name__ == "__main__":
    main()
