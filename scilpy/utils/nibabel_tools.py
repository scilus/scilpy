# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import six


def get_data(nib_file, return_object=False):
    """
    Gets the data array of a Nibabel file or object and clears the cache after
    in order be more memory efficient than the typical img.get_data() function.

    Parameters
    ----------
    nib_file: str or nibabel.nifti object
        Path to the nibabel file to load, or the already-loaded nibabel object.
    return_object: bool
        Whether to return the nibabel object.

    Returns
    -------
    Numpy array of data and the nibabel object if return_object is True.
    """

    if isinstance(nib_file, six.string_types):
        nib_file = nib.load(nib_file)

    # Bypass memmap and unload cached array from memory.
    data = np.array(nib_file.get_data())
    nib_file.uncache()

    if return_object:
        return data, nib_file

    return data
