#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import islice
import os
import six

from dipy.io.streamline import load_tractogram
import nibabel as nib
from nibabel.streamlines import Field, Tractogram
from nibabel.streamlines.trk import (get_affine_rasmm_to_trackvis,
                                     get_affine_trackvis_to_rasmm)


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


def save_from_voxel_space(streamlines, anat, ref_tracts, out_name):
    if isinstance(ref_tracts, six.string_types):
        nib_object = nib.streamlines.load(ref_tracts, lazy_load=True)
    else:
        nib_object = ref_tracts

    if isinstance(anat, six.string_types):
        anat = nib.load(anat)

    affine_to_rasmm = get_affine_trackvis_to_rasmm(nib_object.header)

    tracto = Tractogram(streamlines=streamlines,
                        affine_to_rasmm=affine_to_rasmm)

    spacing = anat.header['pixdim'][1:4]
    tracto.streamlines._data *= spacing

    nib.streamlines.save(tracto, out_name, header=nib_object.header)


def ichunk(sequence, n):
    """ Yield successive n-sized chunks from sequence.

    Parameters
    ----------
    sequence: numpy.ndarray
        streamlines
    n: int
        amount of streamlines to load

    Return
    ------

    chunck: list
        subset of streamlines
    """

    sequence = iter(sequence)
    chunk = list(islice(sequence, n))
    while len(chunk) > 0:
        yield chunk
        chunk = list(islice(sequence, n))


def load_tractogram_with_reference(parser, args, filepath,
                                   bbox_check=True, arg_name=None):

    _, ext = os.path.splitext(filepath)
    if ext == '.trk':
        sft = load_tractogram(filepath, 'same',
                              bbox_valid_check=bbox_check)
    elif ext in ['.tck', '.fib', '.vtk', '.dpy']:
        if arg_name:
            arg_ref = arg_name + '_ref'
            if args.__getattribute__(arg_ref):
                sft = load_tractogram(filepath,
                                      args.__getattribute__(arg_ref),
                                      bbox_valid_check=bbox_check)
            else:
                parser.error('--{} is required for this file format '
                             '{}.'.format(arg_ref, filepath))
        elif args.reference is None:
            parser.error('--reference is required for this file format '
                         '{}.'.format(filepath))

        else:
            sft = load_tractogram(filepath, args.reference,
                                  bbox_valid_check=bbox_check)

    else:
        parser.error('{} is an unsupported file format'.format(filepath))

    return sft
