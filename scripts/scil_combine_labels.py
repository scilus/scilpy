#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import numpy as np
import nibabel as nib
from dipy.io.utils import is_header_compatible
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)

DESCRIPTION = """
    Script to combine labels from multiple volumes,
        if there's overlap, it will overwritten based on the input order.

    >>> se_combine_labels.py animal_labels.nii 20 DKT_labels.nii.gz 44 53\\
            -o labels.nii.gz --out_labels_indices 20 44 53
    """

EPILOG = """
    References:
        [1] Al-Sharif N.B., St-Onge E., Vogel J.W., Theaud G.,
            Evans A.C. and Descoteaux M. OHBM 2019.
            Surface integration for connectome analysis in age prediction.
    """


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('volumes_ids', nargs='+', default=[],
                   help='List of volumes directly followed by their labels:\n'
                        '  Image1 id11 id12  Image2 id21 id22 id23 ... \n'
                        '  "all" can be used instead of id numbers.')

    p.add_argument('-o', '--out_file', required=True,
                   help='Labels volume output.')

    o_ids = p.add_mutually_exclusive_group()
    o_ids.add_argument('--out_labels_ids', type=float, nargs='+', default=[],
                       help='Give a list of labels indices for output images.')

    o_ids.add_argument('--unique', action='store_true',
                       help='Output id with unique labels,'
                            ' excluding first background value.')

    o_ids.add_argument('--group_in_m', action='store_true',
                       help='Add (x*1000000) to each volume labels,'
                            ' where x is the input volume order number.')

    p.add_argument('--background', type=float, default=0.,
                   help='Background id, excluded from output [%(default)s],\n'
                        ' the first one is given as output background value.')
    add_overwrite_arg(p)
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    image_files = []
    indices_per_volume = []

    # Separate argument per volume
    current_indices = None
    used_indices_all = False
    for argument in args.volumes_ids:

        if argument.isdigit():
            current_indices.append(float(argument))
        elif argument.lower() == "all":
            current_indices.append("all")
            used_indices_all = True
        else:
            image_files.append(argument)
            if current_indices is not None:
                indices_per_volume.append(current_indices)
            current_indices = []

    if current_indices is not None:
        indices_per_volume.append(current_indices)

    if len(image_files) != len(indices_per_volume):
        logging.error("No indices was given for a given volume")

    if used_indices_all and args.out_labels_ids:
        logging.error("'All' indices was used with 'out_labels_ids'")

    # Check inputs / output
    assert_inputs_exist(parser, image_files)
    assert_outputs_exist(parser, args, args.out_file)

    # Load volume and do checks
    data_list = []
    first_img = nib.load(image_files[0])
    for i in range(len(image_files)):
        # Load images
        volume_nib = nib.load(image_files[i])
        data = volume_nib.get_fdata()
        data_list.append(data)
        assert (is_header_compatible(first_img, image_files[i]))

        if "all" in indices_per_volume[i]:
            indices_per_volume[i] = np.unique(data)

    filtered_ids_per_vol = []
    # Remove background labels
    for id_list in indices_per_volume:
        new_ids = np.setdiff1d(id_list, args.background)
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
            prefix = i * 1000000
            m_list.append(prefix + np.asarray(filtered_ids_per_vol[i]))
        out_labels = np.hstack(m_list)
    else:
        out_labels = np.hstack(filtered_ids_per_vol)

    if len(np.unique(out_labels)) != len(out_labels):
        logging.error("The same output label number was used "
                      "for multiple inputs")

    # Create the resulting volume
    current_id = 0
    resulting_labels = (np.ones_like(data_list[0], dtype=np.float64)
                        * args.background)
    for i in range(len(image_files)):
        # Add Given labels for each volume
        for index in filtered_ids_per_vol[i]:
            mask = np.isclose(data_list[i], index)
            resulting_labels[mask] = out_labels[current_id]
            current_id += 1

            if np.count_nonzero(mask) == 0:
                logging.warning("Label {} was not in the volume".format(index))

    # Save final combined volume
    nii = nib.Nifti1Image(resulting_labels, first_img.affine, first_img.header)
    nib.save(nii, args.out_file)


if __name__ == "__main__":
    main()
