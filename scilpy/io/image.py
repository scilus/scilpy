# -*- coding: utf-8 -*-

from dipy.io.utils import is_header_compatible
import logging
import nibabel as nib
import numpy as np
import os

from scilpy.utils import is_float


def load_img(arg):
    if is_float(arg):
        img = float(arg)
        dtype = np.float64
    else:
        if not os.path.isfile(arg):
            raise ValueError('Input file {} does not exist.'.format(arg))
        img = nib.load(arg)
        shape = img.header.get_data_shape()
        dtype = img.header.get_data_dtype()
        logging.info('Loaded {} of shape {} and data_type {}.'.format(
                     arg, shape, dtype))

        if len(shape) > 3:
            logging.warning('{} has {} dimensions, be careful.'.format(
                arg, len(shape)))
        elif len(shape) < 3:
            raise ValueError('{} has {} dimensions, not valid.'.format(
                arg, len(shape)))

    return img, dtype


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
                vals = [int(x) for x in filter_opt.split(':')]
                mask[(atlas >= int(min(vals))) & (atlas <= int(max(vals)))] = 1
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
            raise Exception(f"Images are not of the same resolution/affine : "
                            f"({curr_image}) vs ({images[0]})")


def get_data_as_mask(mask_img, dtype=np.uint8):
    """
    Get data as mask (force type np.uint8 or bool), check data type before
    casting.

    Parameters
    ----------
    mask_img: nibabel.nifti1.Nifti1Image
        Mask image.
    dtype: type or str
        Data type for the output data (default: uint8)

    Return
    ------
    data: numpy.ndarray
        Data (dtype : np.uint8 or bool).
    """
    # Verify that out data type is ok
    if not (issubclass(np.dtype(dtype).type, np.uint8) or
            issubclass(np.dtype(dtype).type, np.dtype(bool).type)):
        raise IOError('Output data type must be uint8 or bool. '
                      'Current data type is {}.'.format(dtype))

    # Verify that loaded datatype is ok
    curr_type = mask_img.get_data_dtype().type
    basename = os.path.basename(mask_img.get_filename())
    if np.issubdtype(curr_type, np.signedinteger) or \
        np.issubdtype(curr_type, np.unsignedinteger) \
            or np.issubdtype(curr_type, np.dtype(bool).type):
        data = np.asanyarray(mask_img.dataobj).astype(dtype)

        # Verify that it contains only 0 and 1.
        unique_vals = np.unique(data)
        if len(unique_vals) == 2:
            if np.all(unique_vals != np.array([0, 1])):
                logging.warning('The two unique values in mask were not 0 and'
                                ' 1. Binarizing the mask now.')
                data[data != 0] = 1
        elif len(unique_vals) == 1:
            data[data != 0] = 1
        else:
            raise IOError('The image {} contains more than 2 values. '
                          'It can\'t be loaded as mask.'.format(basename))

    else:
        raise IOError('The image {} cannot be loaded as mask because '
                      'its type {} is not compatible '
                      'with a mask.\n'
                      'To convert your data, you may use tools like mrconvert '
                      'or \n'
                      '>> scil_volume_math.py convert IMG IMG '
                      '--data_type uint8 -f'.format(basename, curr_type))

    return data
