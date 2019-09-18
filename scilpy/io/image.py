#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nibabel as nib


def get_reference_info(reference):
    """Get basic information from a reference NIFTI file.
    :param reference: Nibabel.nifti or filepath (nii or nii.gz)
    """

    # TODO remove/replace after stateful_tractogram PR is merged
    nib_file = nib.load(reference)
    reference_shape = nib_file.get_shape()
    reference_affine = nib_file.affine

    return reference_shape, reference_affine


def assert_same_resolution(*images):
    if len(images) == 0:
        raise Exception("Can't check if images are of the same "
                        "resolution/affine. No image has been given")

    ref = get_reference_info(images[0])
    for i in images[1:]:
        shape, aff = get_reference_info(i)
        if not (ref[0] == shape) and (ref[1] == aff).any():
            raise Exception("Images are not of the same resolution/affine")
