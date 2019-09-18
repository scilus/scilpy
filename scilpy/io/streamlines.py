#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import nibabel as nib


def check_tracts_same_format(parser, tractogram_1, tractogram_2):
    """
    Assert that two filepaths have the same valid extension.
    :param parser: argparse.ArgumentParser object
    :param tractogram_1: Tractogram filename #1
    :param tractogram_2: Tractogram filename #2
    """
    if not nib.streamlines.is_supported(tractogram_1):
        parser.error('Format of "{}" not supported.'.format(tractogram_1))
    if not nib.streamlines.is_supported(tractogram_2):
        parser.error('Format of "{}" not supported.'.format(tractogram_2))

    ext_1 = os.path.splitext(tractogram_1)[1]
    ext_2 = os.path.splitext(tractogram_2)[1]
    if not ext_1 == ext_2:
        parser.error(
            'Input and output tractogram files must use the same format.')


def load_tractogram_with_reference(parser, args, filepath,
                                   bbox_check=True):
    _, ext = os.path.splitext(filepath)
    if ext == '.trk':
        sft = load_tractogram(filepath, 'same',
                              bbox_valid_check=bbox_check)
    elif ext in ['.tck', '.fib', '.vtk', '.dpy']:
        if args.reference is None:
            parser.error('--reference is required for this file format '
                         '{}.'.format(filepath))
        sft = load_tractogram(filepath, args.reference,
                              bbox_valid_check=bbox_check)
    else:
        parser.error('{} is an unsupported file format'.format(filepath))

    return sft
