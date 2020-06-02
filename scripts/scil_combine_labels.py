#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Script to combine labels from multiple volumes,
        if there is overlap, it will overwrite them based on the input order.

    >>> scil_combine_labels.py out_labels.nii.gz  -v animal_labels.nii 20\\
            DKT_labels.nii.gz 44 53  --out_labels_indices 20 44 53
    >>> scil_combine_labels.py slf_labels.nii.gz  -v a2009s_aseg.nii.gz all\\
            -v clean/s1__DKT.nii.gz 1028 2028
"""


import argparse
import logging

from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_label
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


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

    p.add_argument('-v', '--volume_ids', nargs='+', default=[],
                   action='append', required=True,
                   help='List of volumes directly followed by their labels:\n'
                        '  -v atlasA  id1a id2a   -v  atlasB  id1b id2b ... \n'
                        '  "all" can be used instead of id numbers.')

    o_ids = p.add_mutually_exclusive_group()
    o_ids.add_argument('--out_labels_ids', type=int, nargs='+', default=[],
                       help='Give a list of labels indices for output images.')

    o_ids.add_argument('--unique', action='store_true',
                       help='Output id with unique labels,'
                            ' excluding first background value.')

    o_ids.add_argument('--group_in_m', action='store_true',
                       help='Add (x*1000000) to each volume labels,'
                            ' where x is the input volume order number.')

    p.add_argument('--background', type=int, default=0,
                   help='Background id, excluded from output [%(default)s],\n'
                        ' the value is used as output background value.')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image_files = []
    indices_per_volume = []
    # Separate argument per volume
    used_indices_all = False
    for v_args in args.volume_ids:
        if len(v_args) < 2:
            logging.error("No indices was given for a given volume")

        image_files.append(v_args[0])
        if "all" in v_args:
            used_indices_all = True
            indices_per_volume.append("all")
        else:
            indices_per_volume.append(np.asarray(v_args[1:], dtype=np.int))

    if used_indices_all and args.out_labels_ids:
        logging.error("'all' indices cannot be used with 'out_labels_ids'")

    # Check inputs / output
    assert_inputs_exist(parser, image_files)
    assert_outputs_exist(parser, args, args.output)

    # Load volume and do checks
    data_list = []
    first_img = nib.load(image_files[0])
    for i in range(len(image_files)):
        # Load images
        volume_nib = nib.load(image_files[i])
        data = get_data_as_label(volume_nib)
        data_list.append(data)
        assert (is_header_compatible(first_img, image_files[i]))

        if (isinstance(indices_per_volume[i], str)
                and indices_per_volume[i] == "all"):
            indices_per_volume[i] = np.unique(data)

    filtered_ids_per_vol = []
    # Remove background labels
    for id_list in indices_per_volume:
        id_list = np.asarray(id_list)
        new_ids = id_list[~np.in1d(id_list, args.background)]
        filtered_ids_per_vol.append(new_ids)
    # Prepare output indices
    if args.out_labels_ids:
        out_labels = args.out_labels_ids
        if len(out_labels) != len(np.hstack(indices_per_volume)):
            logging.error("--out_labels_ids, requires the same amount"
                          " of total given input indices")
    elif args.unique:
        stack = np.hstack(filtered_ids_per_vol)
        ids = np.arange(len(stack) + 1)
        out_labels = np.setdiff1d(ids, args.background)[:len(stack)]
    elif args.group_in_m:
        m_list = []
        for i in range(len(filtered_ids_per_vol)):
            prefix = i * 10000
            m_list.append(prefix + np.asarray(filtered_ids_per_vol[i]))
        out_labels = np.hstack(m_list)
    else:
        out_labels = np.hstack(filtered_ids_per_vol)

    if len(np.unique(out_labels)) != len(out_labels):
        logging.error("The same output label number was used "
                      "for multiple inputs")

    # Create the resulting volume
    current_id = 0
    resulting_labels = (np.ones_like(data_list[0], dtype=np.uint16)
                        * args.background)
    for i in range(len(image_files)):
        # Add Given labels for each volume
        for index in filtered_ids_per_vol[i]:
            mask = data_list[i] == index
            resulting_labels[mask] = out_labels[current_id]
            current_id += 1

            if np.count_nonzero(mask) == 0:
                logging.warning("Label {} was not in the volume".format(index))

    # Save final combined volume
    nib.save(nib.Nifti1Image(resulting_labels, first_img.affine,
                             header=first_img.header),
             args.output)


if __name__ == "__main__":
    main()
