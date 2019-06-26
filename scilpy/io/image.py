#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scilpy.utils.nibabel_tools import get_data


def get_reference_info(reference):
    """Get basic informations from a reference NIFTI file.
    :param reference: Nibabel.nifti or filepath (nii or nii.gz)
    """

    # TODO remove/replace after stateful_tractogram PR is merged
    _, nib_file = get_data(reference, return_object=True)
    reference_shape = nib_file.get_shape()
    reference_affine = nib_file.affine

    return reference_shape, reference_affine
