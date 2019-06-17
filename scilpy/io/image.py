#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scilpy.utils.nibabel_tools import get_data


def get_reference_info(reference):
    _, nib_file = get_data(reference, return_object=True)
    reference_shape = nib_file.get_shape()
    reference_affine = nib_file.affine

    return reference_shape, reference_affine