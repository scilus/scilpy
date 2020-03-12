#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dipy.io.utils import get_reference_info


def assert_same_resolution(images):
    """
    Check the resolution of multiple images.
    Parameters
    ----------
    images : array of string or string
        List of images or an image.
    """
    if isinstance(images, str):
        images = [images]

    if len(images) == 0:
        raise Exception("Can't check if images are of the same "
                        "resolution/affine. No image has been given")

    aff_1, shape_1, _, _ = get_reference_info(images[0])
    for i in images[1:]:
        aff_2, shape_2, _, _ = get_reference_info(i)
        if not (shape_1 == shape_2) and (aff_1 == aff_2).any():
            raise Exception("Images are not of the same resolution/affine")
