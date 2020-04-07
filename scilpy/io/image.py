# -*- coding: utf-8 -*-

from dipy.io.utils import get_reference_info
import numpy as np


def assert_same_resolution(images):
    """
    Check the resolution of multiple images.
    Parameters
    ----------
    images : array of string or string
        List of images or an image.
    """
    if isinstance(images, str):
        images = [images]

    if len(images) == 0:
        raise Exception("Can't check if images are of the same "
                        "resolution/affine. No image has been given")

    aff_1, shape_1, _, _ = get_reference_info(images[0])
    for i in images[1:]:
        aff_2, shape_2, _, _ = get_reference_info(i)
        if not (shape_1 == shape_2) and (aff_1 == aff_2).any():
            raise Exception("Images are not of the same resolution/affine")


def get_data_as_mask(parser, in_img, filename):
    """
    Get data as mask (force type np.uint8), check data type before casting.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    in_img: nibabel.nifti1.Nifti1Image
        Image
    filename: string
        Filename used for logging.

    Return
    ------
    data: numpy.ndarray
        Data (dtype : np.uint8).
    """

    curr_type = in_img.get_data_dtype().type
    if np.issubdtype(curr_type, np.signedinteger) or np.issubdtype(curr_type, np.unsignedinteger) or np.issubdtype(curr_type, np.bool):
        data = np.asanyarray(in_img.object).astype(np.uint8)
        unique_vals = np.unique(data)
        if len(unique_vals) == 2:
            if np.all(unique_vals != np.array[0, 1]):
                data[data != 0] = 1
        else:
            parser.error('The image {} contains more than 2 values. '
                         'It can\t be loaded as mask.'.format(filename))

    else:
        parser.error('The image {} cannot be loaded as mask because '
                     'its type {} is not compatible '
                     'with a mask'.format(filename, curr_type))

    return data


def get_data_as_label(parser, in_img, filename):
    """
    Get data as label (force type np.uint16), check data type before casting.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    in_img: nibabel.nifti1.Nifti1Image
        Image.
    filename: string
        Filename used for logging.

    Return
    ------

    data: numpy.ndarray
        Data (dtype: np.uint16).
    """

    curr_type = in_img.get_data_dtype()
    if np.issubdtype(curr_type, np.int):
        return np.asanyarray(in_img.object).astype(np.uint16)
    else:
        parser.error('The image {} cannot be loaded as label because ')
