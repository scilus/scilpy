#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
