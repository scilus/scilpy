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

    data = nb_object.get_fdata(dtype=np.float32, caching='unchanged')

    # Count the number of non-zero voxels.
    if len(data.shape) >= 4:
        axes_to_sum = np.arange(3, len(data.shape))
        nb_voxels = np.count_nonzero(np.sum(np.absolute(data),
                                            axis=tuple(axes_to_sum)))
    else:
        nb_voxels = np.count_nonzero(data)

    return nb_voxels


def volume_iterator(img, blocksize=1, start=0, end=0):
    """Generator that iterates on volumes of data.

    Parameters
    ----------
    img : nib.Nifti1Image
        Image of a 4D volume with shape X,Y,Z,N
    blocksize : int, optional
        Number of volumes to return in a single batch
    start : int, optional
        Starting iteration index in the 4D volume
    end : int, optional
        Stopping iteration index in the 4D volume
        (the volume at this index is excluded)

    Yields
    -------
    tuple of (list of int, ndarray)
        The ids of the selected volumes, and the selected data as a 4D array
    """
    assert end <= img.shape[-1], "End limit provided is greater than the " \
                                 "total number of volumes in image"

    nb_volumes = img.shape[-1]
    end = end if end else img.shape[-1]

    if blocksize == nb_volumes:
        yield list(range(start, end)), \
              img.get_fdata(dtype=np.float32)[..., start:end]
    else:
        stop = start
        for i in range(start, end - blocksize, blocksize):
            start, stop = i, i + blocksize
            logging.info("Loading volumes {} to {}.".format(start, stop - 1))
            yield list(range(start, stop)), img.dataobj[..., start:stop]

        if stop < end:
            logging.info(
                "Loading volumes {} to {}.".format(stop, end - 1))
            yield list(range(stop, end)), img.dataobj[..., stop:end]
