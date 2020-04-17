# -*- coding: utf-8 -*-

from dipy.io.utils import is_header_compatible


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

    for i in images[1:]:
        if not is_header_compatible(images[0], images[1]):
            raise Exception("Images are not of the same resolution/affine")
