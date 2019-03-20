#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path


def split_name_with_nii(filename):
    """
    Returns the clean basename and extension of a file.
    Means that this correctly manages the ".nii.gz" extensions.
    :param filename: The filename to clean
    :return: A tuple of the clean basename and the full extension
    """
    base, ext = os.path.splitext(filename)

    if ext == ".gz":
        # Test if we have a .nii additional extension
        temp_base, add_ext = os.path.splitext(base)

        if add_ext == ".nii":
            ext = add_ext + ext
            base = temp_base

    return base, ext
