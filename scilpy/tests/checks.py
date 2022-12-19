import numpy as np


def assert_images_close(img1, img2):
    dtype = img1.header.get_data_dtype()

    assert np.allclose(img1.affine, img2.affine), "Images affines don't match"

    assert np.allclose(
        img1.get_fdata(dtype=dtype), img2.get_fdata(dtype=dtype)), \
        "Images data don't match. MSE : {} | max SE : {}".format(
            np.mean((img1.get_fdata(dtype=dtype) -
                     img2.get_fdata(dtype=dtype)) ** 2.),
            np.max((img1.get_fdata(dtype=dtype) -
                    img2.get_fdata(dtype=dtype)) ** 2.))
