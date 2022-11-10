# -*- coding: utf-8 -*-
import os

import numpy as np


def get_data_as_labels(in_img):
    """
    Get data as label (force type np.uint16), check data type before casting.

    Parameters
    ----------
    in_img: nibabel.nifti1.Nifti1Image
        Image.

    Return
    ------
    data: numpy.ndarray
        Data (dtype: np.uint16).
    """
    curr_type = in_img.get_data_dtype()

    if np.issubdtype(curr_type, np.signedinteger) or \
       np.issubdtype(curr_type, np.unsignedinteger):
        return np.asanyarray(in_img.dataobj).astype(np.uint16)
    else:
        basename = os.path.basename(in_img.get_filename())
        raise IOError('The image {} cannot be loaded as label because '
                      'its format {} is not compatible with a label '
                      'image'.format(basename, curr_type))
