# -*- coding: utf-8 -*-

import logging

import nibabel as nib
import numpy as np
import six


def count_non_zero_voxels(image):
    """
    Count number of non zero voxels

    Parameters:
    -----------
    image: string
        Path to the image
    """
    if isinstance(image, six.string_types):
        nb_object = nib.load(image)
    else:
        nb_object = image

    data = nb_object.get_data(caching='unchanged')

    # Count the number of non-zero voxels.
    if len(data.shape) >= 4:
        axes_to_sum = np.arange(3, len(data.shape))
        nb_voxels = np.count_nonzero(np.sum(np.absolute(data),
                                            axis=tuple(axes_to_sum)))
    else:
        nb_voxels = np.count_nonzero(data)

    return nb_voxels


def volume_iterator(img, blocksize=1):
    """Generator that iterates on volumes of data.

    Parameters
    ----------
    img : nib.Nifti1Image
        Image of a 4D volume with shape X,Y,Z,N
    blocksize : int, optional
        Number of volumes to return in a single batch

    Yields
    -------
    tuple of (list of int, ndarray)
        The ids of the selected volumes, and the selected data as a 4D array
    """
    nb_volumes = img.shape[-1]

    if blocksize == nb_volumes:
        yield list(range(nb_volumes)), img.get_fdata(dtype=np.float32)
    else:
        start, end = 0, 0
        for i in range(0, nb_volumes - blocksize, blocksize):
            start, end = i, i + blocksize
            logging.info("Loading volumes {} to {}.".format(start, end - 1))
            yield list(range(start, end)), img.dataobj[..., start:end]

        if end < nb_volumes:
            logging.info(
                "Loading volumes {} to {}.".format(end, nb_volumes - 1))
            yield list(range(end, nb_volumes)), img.dataobj[..., end:]
