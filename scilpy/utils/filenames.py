# -*- coding: utf-8 -*-

import os.path


def add_filename_suffix(filename, suffix):
    """
    This function adds a suffix to the filename, keeping the extension.
    For example, if filename is test.nii.gz and suffix is "new",
    the returned name will be test_new.nii.gz

    Parameters
    ----------
    filename: str
        The full filename, including extension
    suffix: str
        The suffix to add to the filename

    Returns
    -------
    The completed file name.
    """
    base, ext = split_name_with_nii(filename)

    return base + suffix + ext


def split_name_with_nii(filename):
    """
    Returns the clean basename and extension of a file.
    Means that this correctly manages the ".nii.gz" extensions.

    Parameters
    ----------
    filename: str
        The filename to clean

    Returns
    -------
        base, ext : tuple(str, str)
        Clean basename and the full extension
    """
    base, ext = os.path.splitext(filename)

    if ext == ".gz":
        # Test if we have a .nii additional extension
        temp_base, add_ext = os.path.splitext(base)

        if add_ext == ".nii":
            ext = add_ext + ext
            base = temp_base

    return base, ext
