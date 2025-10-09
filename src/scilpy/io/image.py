# -*- coding: utf-8 -*-

from dipy.io.utils import is_header_compatible
import logging
import nibabel as nib
import numpy as np
import os

from scilpy.utils import is_float


def load_nifti_reorient(file_path, return_flip_vector=False):
    vol = nib.load(file_path)

    # Compute the image orientation (axis codes)
    axcodes = nib.orientations.aff2axcodes(vol.affine)
    target = ('R', 'A', 'S')
    flip_vector = [1 if axcodes[i] == target[i] else -1 for i in range(3)]

    ras_order = [[i, flip_vector[i]] for i in range(0, 3)]
    filename = vol.get_filename()
    vol = vol.as_reoriented(ras_order)
    vol.set_filename(filename)

    if return_flip_vector:
        return vol, flip_vector
    return vol


def nifti_reorient(img, flip_vector):
    original_order = [[i, flip_vector[i]] for i in range(0, 3)]
    img = img.as_reoriented(original_order)
    return img


def save_nifti_reorient(img, flip_vector, file_path):
    original_order = [[i, flip_vector[i]] for i in range(0, 3)]
    img = img.as_reoriented(original_order)
    nib.save(img, file_path)


def load_img(arg):
    """
    Function to create the variable for scil_volume_math main function.
    It can be a float or an image and if image it checks if it contains
    integer values and its declared data type is integer or if it is containing
    float values but declared as integer, in which case a warning is raised.
    Parameters
    """
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
        data_as_float = img.get_fdata()
        sum_float = float(np.sum(data_as_float))

        if not sum_float.is_integer():
            logging.warning('Image {} has an integer type but contains '
                            'non-integer values. Loading, computating and saving '
                            'will be done as float. Using an integer dtype '
                            'will lead to data loss.'.format(arg))
            dtype = np.float64
            img.header.set_data_dtype(dtype)

        if len(shape) > 3:
            logging.warning('{} has {} dimensions, be careful.'.format(
                arg, len(shape)))
        elif len(shape) < 3:
            raise ValueError('{} has {} dimensions, not valid.'.format(
                arg, len(shape)))

    return img, dtype


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
