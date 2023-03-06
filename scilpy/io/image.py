# -*- coding: utf-8 -*-

from dipy.io.utils import is_header_compatible
import logging
import numpy as np
import os


def merge_labels_into_mask(atlas, filtering_args):
    """
    Merge labels into a mask.

    Parameters
    ----------
    atlas: np.ndarray
        Atlas with labels as a numpy array (uint16) to merge.

    filtering_args: str
        Filtering arguments from the command line.

    Return
    ------
    mask: nibabel.nifti1.Nifti1Image
        Mask obtained from the combination of multiple labels.
    """
    mask = np.zeros(atlas.shape, dtype=np.uint16)

    if ' ' in filtering_args:
        values = filtering_args.split(' ')
        for filter_opt in values:
            if ':' in filter_opt:
                values = [int(x) for x in filter_opt.split(':')]
                mask[(atlas >= int(min(values))) & (atlas <= int(max(values)))] = 1
            else:
                mask[atlas == int(filter_opt)] = 1
    elif ':' in filtering_args:
        values = [int(x) for x in filtering_args.split(':')]
        mask[(atlas >= int(min(values))) & (atlas <= int(max(values)))] = 1
    else:
        mask[atlas == int(filtering_args)] = 1

    return mask


def assert_same_resolution(images):
    """
    Check the resolution of multiple images.
    Parameters
    ----------
    images : list of string or string
        List of images or an image.
    """
    if isinstance(images, str):
        images = [images]

    if len(images) == 0:
        raise Exception("Can't check if images are of the same "
                        "resolution/affine. No image has been given")

    for curr_image in images[1:]:
        if not is_header_compatible(images[0], curr_image):
            raise Exception("Images are not of the same resolution/affine")


def get_data_as_mask(in_img, dtype=np.uint8):
    """
    Get data as mask (force type np.uint8 or bool), check data type before
    casting.

    Parameters
    ----------
    in_img: nibabel.nifti1.Nifti1Image
        Image

    dtype: data type for the output data (default: uint8)
        type

    Return
    ------
    data: numpy.ndarray
        Data (dtype : np.uint8 or bool).
    """
    if not (issubclass(np.dtype(dtype).type, np.uint8) or
            issubclass(np.dtype(dtype).type, np.dtype(bool).type)):
        raise IOError('Output data type must be uint8 or bool. '
                      'Current data type is {}.'.format(dtype))

    curr_type = in_img.get_data_dtype().type
    basename = os.path.basename(in_img.get_filename())
    if np.issubdtype(curr_type, np.signedinteger) or \
        np.issubdtype(curr_type, np.unsignedinteger) \
            or np.issubdtype(curr_type, np.dtype(bool).type):
        data = np.asanyarray(in_img.dataobj).astype(dtype)
        unique_vals = np.unique(data)
        if len(unique_vals) == 2:
            if np.all(unique_vals != np.array([0, 1])):
                logging.warning('The two unique values in mask were not 0 and'
                                ' 1. Tha mask has been binarised.')
                data[data != 0] = 1
        elif len(unique_vals) == 1:
            data[data != 0] = 1
        else:
            raise IOError('The image {} contains more than 2 values. '
                          'It can\'t be loaded as mask.'.format(basename))

    else:
        raise IOError('The image {} cannot be loaded as mask because '
                      'its type {} is not compatible '
                      'with a mask'.format(basename, curr_type))

    return data
