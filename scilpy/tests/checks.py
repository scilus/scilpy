# -*- coding: utf-8 -*-

import numpy as np


def _nd_array_match(_arr1, _arr2, _rtol=1E-05, _atol=1E-8):
    return np.allclose(_arr1, _arr2, rtol=_rtol, atol=_atol)


def _mse_metrics(_arr1, _arr2):
    _mse = (_arr1 - _arr2) ** 2.
    return np.mean(_mse), np.max(_mse)


def assert_images_close(img1, img2):
    dtype = img1.header.get_data_dtype()

    assert np.allclose(img1.affine, img2.affine), "Images affines don't match"

    assert _nd_array_match(img1.get_fdata(dtype=dtype),
                           img2.get_fdata(dtype=dtype)), \
        "Images data don't match. MSE : {} | max SE : {}".format(
            *_mse_metrics(img1.get_fdata(dtype=dtype),
                          img2.get_fdata(dtype=dtype)))



def assert_images_not_close(img1, img2, affine_must_match=True):
    dtype = img1.header.get_data_dtype()

    if affine_must_match:
        assert np.allclose(img1.affine, img2.affine), \
            "Images affines don't match"

    assert not _nd_array_match(img1.get_fdata(dtype=dtype),
                           img2.get_fdata(dtype=dtype)), \
        "Images data should not match. MSE : {} | max SE : {}".format(
            *_mse_metrics(img1.get_fdata(dtype=dtype),
                          img2.get_fdata(dtype=dtype)))
