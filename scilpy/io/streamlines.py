#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import islice
import os
import six

from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
import nibabel as nib
from nibabel.streamlines import Field, Tractogram
from nibabel.streamlines.trk import (get_affine_rasmm_to_trackvis,
                                     get_affine_trackvis_to_rasmm)
import numpy as np


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


def load_trk_in_voxel_space(trk_file, anat=None, grid_res=None,
                            raise_on_empty=True):
    """
    Load streamlines in voxel space, corner aligned

    :param trk_file: path or nibabel object
    :param anat: path or nibabel image (optional)
    :param grid_res: specify the grid resolution (3,) (optional)
    :param raise_on_empty: specify if an error should be raised when
           the file is empty
    :return: NiBabel tractogram with streamlines loaded in voxel space
    """
    if isinstance(trk_file, six.string_types):
        trk_file = nib.streamlines.load(trk_file)

    if trk_file.header[Field.NB_STREAMLINES] == 0 and raise_on_empty:
        raise Exception("The file contains no streamline")

    if anat and grid_res:
        raise Exception("Parameters anat and grid_res cannot be used together")

    if anat:
        if isinstance(anat, six.string_types):
            anat = nib.load(anat)
        spacing = anat.header['pixdim'][1:4]
    elif grid_res:
        spacing = grid_res
    else:
        spacing = trk_file.header[Field.VOXEL_SIZES]

    affine_to_voxmm = get_affine_rasmm_to_trackvis(trk_file.header)
    tracto = trk_file.tractogram
    tracto.apply_affine(affine_to_voxmm)
    streamlines = tracto.streamlines

    if trk_file.header[Field.NB_STREAMLINES] > 0:
        streamlines._data /= spacing
    return streamlines


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


def filter_tractogram_data(tractogram, streamline_ids):
    """ Filter tractogram according to streamline ids and keep
    the data

    Parameters:
    -----------
    tractogram: Tractogram or StatefulTractogram
        Tractogram containing the data to be filtered
    streamline_ids: array_like
        List of streamline ids the data corresponds to

    Returns:
    --------
    new_tractogram: Tractogram or StatefulTractogram
        Returns the same type as the input but only with
    """

    assert isinstance(tractogram, Tractogram) or \
        isinstance(tractogram, StatefulTractogram), \
        ("Tractogram must either be Tracrogram or StatefulTractogram")

    streamline_ids = np.asarray(streamline_ids, dtype=np.int)

    assert np.all(
        np.in1d(streamline_ids, np.arange(len(tractogram.streamlines)))
    ), "Received ids outside of streamline range"

    new_streamlines = tractogram.streamlines[streamline_ids]
    new_data_per_streamline = tractogram.data_per_streamline[streamline_ids]
    new_data_per_point = tractogram.data_per_point[streamline_ids]

    # Could have been nice to deepcopy the tractogram modify the attributes in
    # place instead of creating a new one, but tractograms cant be subsampled
    # if they have data
    if isinstance(tractogram, Tractogram):
        return Tractogram(
            new_streamlines,
            affine_to_rasmm=tractogram.affine_to_rasmm,
            data_per_point=new_data_per_point,
            data_per_streamline=new_data_per_streamline)

    return StatefulTractogram(
        new_streamlines,
        tractogram,
        tractogram.space,
        data_per_point=new_data_per_point,
        data_per_streamline=new_data_per_streamline)
