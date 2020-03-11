#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Warp *.trk using a non linear deformation.
    Can be used with Ants or Dipy deformation map.

    For more information on how to use the various registration scripts
    see the doc/tractogram_registration.md readme file
"""

import argparse

from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, add_reference_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.utils.streamlines import warp_streamlines


def _build_args_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('moving_tractogram',
                   help='Path of the tractogram to be transformed.')
    p.add_argument('target_file',
                   help='Path of the reference file (trk or nii).')
    p.add_argument('deformation',
                   help='Path of the file containing deformation field.')

    p.add_argument('out_tractogram',
                   help='Output filename of the transformed tractogram.')

    p.add_argument('--field_source', default='ants', choices=['ants', 'dipy'],
                   help='Source of the deformation field: [%(choices)s]  \n'
                        'be cautious, the default is  [%(default)s].')

    add_overwrite_arg(p)
    add_reference_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.moving_tractogram, args.target_file,
                                 args.deformation])
    assert_outputs_exist(parser, args, args.out_tractogram)

    sft = load_tractogram_with_reference(parser, args, args.moving_tractogram)

    deformation = nib.load(args.deformation)
    deformation_data = np.squeeze(deformation.get_fdata())

    if not is_header_compatible(sft, deformation):
        parser.error('Input tractogram/reference do not have the same spatial '
                     'attribute as the deformation field.')

    # Warning: Apply warp in-place
    moved_streamlines = warp_streamlines(sft, deformation_data,
                                         args.field_source)
    new_sft = StatefulTractogram(moved_streamlines, args.target_file,
                                 Space.RASMM,
                                 data_per_point=sft.data_per_point,
                                 data_per_streamline=sft.data_per_streamline)
    save_tractogram(new_sft, args.out_tractogram)


if __name__ == "__main__":
    main()
