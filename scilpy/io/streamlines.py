# -*- coding: utf-8 -*-

from itertools import islice
import logging
import os
import tempfile

from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram as _save_tractogram
from dipy.io.utils import is_header_compatible
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np

from scilpy.io.utils import load_matrix_in_any_format


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

    chunk: list
        subset of streamlines
    """

    sequence = iter(sequence)
    chunk = list(islice(sequence, n))
    while len(chunk) > 0:
        yield chunk
        chunk = list(islice(sequence, n))


def is_argument_set(args, arg_name):
    # Check that attribute is not None
    return not getattr(args, arg_name, None) is None


def load_tractogram_with_reference(parser, args, filepath, arg_name=None):
    """
    Parameters
    ----------
    parser: Argument Parser
        Used to print errors, if any.
    args: Namespace
        Parsed arguments. Used to get the 'reference' and 'bbox_check' args.
        See scilpy.io.utils to add the arguments to your parser.
    filepath: str
        Path of the tractogram file.
    arg_name: str, optional
        Name of the reference argument. By default the args.ref is used. If
        arg_name is given, then args.arg_name_ref will be used instead.
    """
    if is_argument_set(args, 'bbox_check'):
        bbox_check = args.bbox_check
    else:
        bbox_check = True

    _, ext = os.path.splitext(filepath)
    if ext == '.trk':
        if (is_argument_set(args, 'reference') or
                arg_name and args.__getattribute__(arg_name + '_ref')):
            logging.warning('Reference is discarded for this file format '
                            '{}.'.format(filepath))
        sft = load_tractogram(filepath, 'same',
                              bbox_valid_check=bbox_check)

        # Force dtype to int64 instead of float64
        if len(sft.streamlines) == 0:
            sft.streamlines._offsets.dtype = np.dtype(np.int64)

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
        elif (not is_argument_set(args, 'reference')) or args.reference is None:
            parser.error('--reference is required for this file format '
                         '{}.'.format(filepath))
        else:
            sft = load_tractogram(filepath, args.reference,
                                  bbox_valid_check=bbox_check)

    else:
        parser.error('{} is an unsupported file format'.format(filepath))

    return sft


def save_tractogram(sft, filename, no_empty, bbox_valid_check=True):
    if len(sft.streamlines) == 0 and no_empty:
        logging.info("The file {} won't be written (0 streamlines)"
                     .format(filename))
    else:
        if len(sft.streamlines) == 0:
            logging.info("Writing an empty file (0 streamlines): {} "
                         .format(filename))
        _save_tractogram(sft, filename, bbox_valid_check=bbox_valid_check)


def verify_compatibility_with_reference_sft(ref_sft, files_to_verify,
                                            parser, args):
    """
    Verifies the compatibility of a reference sft with a list of files.

    Params
    ------
    ref_sft: StatefulTractogram
        A tractogram to be used as reference.
    files_to_verify: List[str]
        List of files that should be compatible with the reference sft. Files
        can be either other tractograms or nifti files (ex: masks).
    parser: argument parser
        Will raise an error if a file is not compatible.
    args: Namespace
        Should contain a args.reference if any file is a .tck, and possibly a
        args.bbox_check (set to True by default).
    """
    save_ref = args.reference

    for file in files_to_verify:
        if file is not None:
            _, ext = os.path.splitext(file)
            if ext in ['.trk', '.tck', '.fib', '.vtk', '.dpy']:
                # Cheating ref because it may send a lot of warning if loading
                # many trk with ref (reference was maybe added only for some
                # of these files)
                if ext == '.trk':
                    args.reference = None
                else:
                    args.reference = save_ref
                mask = load_tractogram_with_reference(parser, args, file)
            else:  # should be a nifti file.
                mask = file
            compatible = is_header_compatible(ref_sft, mask)
            if not compatible:
                parser.error("Reference tractogram incompatible with {}"
                             .format(file))


def load_dps_files_as_dps(parser, dps_files, sft, keys=None, overwrite=False):
    """
    Load dps information. They must be scalar values.

    Parameters
    ----------
    parser: parser
    dps_files: list[str]
        Either .npy or .txt files.
    sft: StatefulTractogram
    keys: list[str]
        If None, use the filenames as keys.
    overwrite: bool
        If True, allow overwriting an existing dps key.

    Returns
    -------
    sft: StatefulTractogram
        The modified SFT. (Note that it is modified in-place even if the
        returned variable is not used!)
    new_keys: list[str]
        Added keys.
    """
    if keys is not None and len(keys) != len(dps_files):
        parser.error("You must provide one key name per dps file.")

    new_keys = []
    for i, file in enumerate(dps_files):
        if keys is None:
            name = os.path.basename(file)
            key, ext = os.path.splitext(name)
        else:
            key = keys[i]

        if key in sft.data_per_streamline and not overwrite:
            parser.error("Key {} already exists in your tractogram's dps. "
                         "You must allow overwriting keys."
                         .format(key))

        data = np.squeeze(load_matrix_in_any_format(file))
        if len(data) != len(sft):
            parser.error('Wrong dps size in file {}. Expected one value per '
                         'streamline ({}) but got {} values!'
                         .format(file, len(sft), len(data)))

        new_keys.append(key)
        sft.data_per_streamline[key] = data
    return sft, new_keys


def load_dpp_files_as_dpp(parser, dpp_files, sft, keys=None, overwrite=False):
    """
    Load dpp information. They must be scalar values.

    Parameters
    ----------
    parser: parser
    dpp_files: list[str]
        Either .npy or .txt files.
    sft: StatefulTractogram
    keys: list[str]
        If None, use the filenames as keys.
    overwrite: bool
        If True, allow overwriting an existing dpp key.

    Returns
    -------
    sft: StatefulTractogram
        The modified SFT. (Note that it is modified in-place even if the
        returned variable is not used!)
    new_keys: list[str]
        Added keys.
    """
    if keys is not None and len(keys) != len(dpp_files):
        parser.error("You must provide one key name per dps file.")

    new_keys = []
    for i, file in enumerate(dpp_files):
        if keys is None:
            name = os.path.basename(file)
            key, ext = os.path.splitext(name)
        else:
            key = keys[i]

        if key in sft.data_per_streamline and not overwrite:
            parser.error("Key {} already exists in your tractogram's dpp. "
                         "You must allow overwriting keys."
                         .format(key))

        data = np.squeeze(load_matrix_in_any_format(file))
        if len(data) != len(sft.streamlines._data):
            parser.error('Wrong dpp size in file {}. Expected one value per '
                         'point in your tractogram ({}) but got {}!'
                         .format(file, len(sft.streamlines._data), len(data)))
        new_keys.append(key)
        sft.data_per_point[key] = data
    return sft, new_keys


def streamlines_to_memmap(input_streamlines,
                          strs_dtype='float32'):
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
    data = np.memmap(data_filename, dtype=strs_dtype, mode='w+',
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


def reconstruct_streamlines_from_memmap(memmap_filenames, indices=None,
                                        strs_dtype='float32'):
    """
    Function to reconstruct streamlines from memmaps, mainly to facilitate
    multiprocessing and decrease RAM usage.

    Parameters
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

    data = np.memmap(memmap_filenames[0],  dtype=strs_dtype, mode='r')
    offsets = np.memmap(memmap_filenames[1],  dtype='int64', mode='r')
    lengths = np.memmap(memmap_filenames[2],  dtype='int32', mode='r')

    return reconstruct_streamlines(data, offsets, lengths, indices=indices)


def reconstruct_streamlines(data, offsets, lengths, indices=None):
    """
    Function to reconstruct streamlines from its data, offsets and lengths
    (from the nibabel tractogram object).

    Parameters
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

    if data.ndim == 2:
        data = np.array(data).flatten()

    if indices is None:
        indices = np.arange(len(offsets))

    streamlines = []
    for i in indices:
        streamline = data[offsets[i]*3:offsets[i]*3 + lengths[i]*3]
        streamlines.append(streamline.reshape((lengths[i], 3)))

    return ArraySequence(streamlines)
