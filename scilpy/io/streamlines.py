# -*- coding: utf-8 -*-

from itertools import islice
import os
import tempfile

from dipy.io.streamline import load_tractogram
import h5py
import nibabel as nib
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


def lazy_streamlines_count(in_tractogram_path):
    """ Gets the number of streamlines as written in the tractogram header.

    Parameters
    ----------
    in_tractogram_path: str
        Tractogram filepath, must be .trk or .tck.

    Return
    ------
    count: int
        Number of streamlines present in the tractogram.
    """
    _, ext = os.path.splitext(in_tractogram_path)
    if ext == '.trk':
        key = 'nb_streamlines'
    elif ext == '.tck':
        key = 'count'
    else:
        raise IOError('{} is not supported for lazy loading'.format(ext))

    tractogram_file = nib.streamlines.load(in_tractogram_path,
                                           lazy_load=True)
    return tractogram_file.header[key]


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


def streamlines_to_memmap(input_streamlines):
    """
    Function to decompose on disk the array_sequence into its components.
    Parameters
    ----------
    input_streamlines : ArraySequence
        All streamlines of the tractogram to segment.
    Returns
    -------
    tmp_obj : tuple
        Temporary directory and tuple of filenames for the data, offsets
        and lengths.
    """
    tmp_dir = tempfile.TemporaryDirectory()
    data_filename = os.path.join(tmp_dir.name, 'data.dat')
    data = np.memmap(data_filename, dtype='float32', mode='w+',
                     shape=input_streamlines._data.shape)
    data[:] = input_streamlines._data[:]

    offsets_filename = os.path.join(tmp_dir.name, 'offsets.dat')
    offsets = np.memmap(offsets_filename, dtype='int64', mode='w+',
                        shape=input_streamlines._offsets.shape)
    offsets[:] = input_streamlines._offsets[:]

    lengths_filename = os.path.join(tmp_dir.name, 'lengths.dat')
    lengths = np.memmap(lengths_filename, dtype='int32', mode='w+',
                        shape=input_streamlines._lengths.shape)
    lengths[:] = input_streamlines._lengths[:]

    return tmp_dir, (data_filename, offsets_filename, lengths_filename)


def reconstruct_streamlines_from_memmap(memmap_filenames, indices=None):
    """
    Function to reconstruct streamlines from memmaps, mainly to facilitate
    multiprocessing and decrease RAM usage.

    ----------
    memmap_filenames : tuple
        Tuple of 3 filepath to numpy memmap (data, offsets, lengths).
    indices : list
        List of int representing the indices to reconstruct.

    Returns
    -------
    streamlines : list of np.ndarray
        List of streamlines.
    """

    data = np.memmap(memmap_filenames[0],  dtype='float32', mode='r')
    offsets = np.memmap(memmap_filenames[1],  dtype='int64', mode='r')
    lengths = np.memmap(memmap_filenames[2],  dtype='int32', mode='r')

    return reconstruct_streamlines(data, offsets, lengths, indices=indices)


def reconstruct_streamlines_from_hdf5(hdf5_filename, key=None):
    """
    Function to reconstruct streamlines from hdf5, mainly to facilitate
    decomposition into thousand of connections and decrease I/O usage.
    ----------
    hdf5_filename : str
        Filepath to the hdf5 file.
    key : str
        Key of the connection of interest (LABEL1_LABEL2).

    Returns
    -------
    streamlines : list of np.ndarray
        List of streamlines.
    """

    hdf5_file = hdf5_filename

    if key is not None:
        if key not in hdf5_file:
            return []
        group = hdf5_file[key]
    else:
        group = hdf5_file

    data = np.array(group['data']).flatten()
    offsets = np.array(group['offsets'])
    lengths = np.array(group['lengths'])

    return reconstruct_streamlines(data, offsets, lengths)


def reconstruct_streamlines(data, offsets, lengths, indices=None):
    """
    Function to reconstruct streamlines from its data, offsets and lengths
    (from the nibabel tractogram object).

    ----------
    data : np.ndarray
        Nx3 array representing all points of the streamlines.
    offsets : np.ndarray
        Nx1 array representing the cumsum of length array.
    lengths : np.ndarray
        Nx1 array representing the length of each streamline.
    indices : list
        List of int representing the indices to reconstruct.

    Returns
    -------
    streamlines : list of np.ndarray
        List of streamlines.
    """

    if indices is None:
        indices = np.arange(len(offsets))

    streamlines = []
    for i in indices:
        streamline = data[offsets[i]*3:offsets[i]*3 + lengths[i]*3]
        streamlines.append(streamline.reshape((lengths[i], 3)))

    return streamlines
