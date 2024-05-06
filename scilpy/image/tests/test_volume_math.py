# -*- coding: utf-8 -*-
import nibabel as nib
import numpy as np
from numpy.testing import (assert_array_equal,
                           assert_allclose,
                           assert_array_almost_equal)
from scipy.ndimage import (binary_closing, binary_dilation,
                           binary_erosion, binary_opening,
                           gaussian_filter)


from scilpy.image.volume_math import (_validate_imgs_type,
                                      _validate_length,
                                      _validate_type,
                                      _validate_float,
                                      _get_neighbors,
                                      lower_threshold_eq, upper_threshold_eq,
                                      lower_threshold, upper_threshold,
                                      lower_threshold_otsu,
                                      upper_threshold_otsu,
                                      lower_clip, upper_clip,
                                      absolute_value,
                                      around, ceil, floor,
                                      normalize_max, normalize_sum,
                                      base_10_log, natural_log,
                                      convert,
                                      addition, subtraction,
                                      multiplication, division,
                                      mean, std,
                                      union, intersection,
                                      difference, invert,
                                      concatenate, gaussian_blur,
                                      dilation, erosion,
                                      closing, opening,
                                      neighborhood_correlation)


EPSILON = np.finfo(float).eps


def _assert_failed(function, *args):
    """
    Assert that a call fails
    """
    failed = False
    try:
        function(*args)
    except ValueError:  # All our tests raise ValueErrors.
        failed = True
    assert failed


def test_validate_imgs_type():
    fake_affine = np.eye(4)
    data1 = np.zeros((5, 5, 5)).astype(float)
    img1 = nib.Nifti1Image(data1, fake_affine)

    # 1) If they are all images
    data2 = np.zeros((6, 6, 6)).astype(float)
    img2 = nib.Nifti1Image(data2, fake_affine)
    _validate_imgs_type(img1, img2)  # Should pass

    # 2) If one input is not an image
    _assert_failed(_validate_imgs_type, img1, data2)


def test_validate_length():

    # 1) Exact length
    exact_length_list = [1, 2, 3]
    _validate_length(exact_length_list, 3)  # Should pass

    shorter_list = [1, 2]
    _assert_failed(_validate_length, shorter_list, 3)

    # 2) With option 'at_least'
    at_least_list = [1, 2, 3, 4]
    _validate_length(at_least_list, 3, at_least=True)  # Should pass

    shorter_list = [1, 2]
    _assert_failed(_validate_length, shorter_list, 3)


def test_validate_type():

    correct_type_input = 42  # Integer
    _validate_type(correct_type_input, int)  # Should pass

    incorrect_type_input = "42"  # String
    _assert_failed(_validate_type, incorrect_type_input, int)


def test_validate_float():

    correct_input = 42  # Integer can be cast to float
    _validate_float(correct_input)  # Should pass

    incorrect_input = "not_a_float"  # String that can't be cast to float
    _assert_failed(_validate_float, incorrect_input)


def test_lower_threshold_eq():
    # Create a sample nib.Nifti1Image object
    img_data = np.array([0, 1, 2, 3, 4, 5]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)

    # Define a threshold value
    threshold = 3

    # Expected output after lower thresholding
    expected_output = np.array([0, 0, 0, 1, 1, 1])

    # Run the function
    output_data = lower_threshold_eq([img, threshold], img)

    # Assert that the output matches the expected output
    assert_array_equal(output_data, expected_output)


def test_upper_threshold_eq():
    # Create a sample nib.Nifti1Image object
    img_data = np.array([0, 1, 2, 3, 4, 5]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)

    # Define a threshold value
    threshold = 3

    # Expected output after upper thresholding
    expected_output = np.array([1, 1, 1, 1, 0, 0])

    # Run the function
    output_data = upper_threshold_eq([img, threshold], img)

    # Assert that the output matches the expected output
    assert_array_equal(output_data, expected_output)


def test_lower_threshold():
    # Create a sample nib.Nifti1Image object
    img_data = np.array([0, 1, 2, 3, 4, 5]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)

    # Define a threshold value
    threshold = 3

    # Expected output after lower thresholding
    expected_output = np.array([0, 0, 0, 0, 1, 1]).astype(float)

    # Run the function
    output_data = lower_threshold([img, threshold], img)

    # Assert that the output matches the expected output
    assert_array_equal(output_data, expected_output)


def test_upper_threshold():
    # Create a sample nib.Nifti1Image object
    img_data = np.array([0, 1, 2, 3, 4, 5]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)

    # Define a threshold value
    threshold = 3

    # Expected output after upper thresholding
    expected_output = np.array([1, 1, 1, 0, 0, 0]).astype(float)

    # Run the function
    output_data = upper_threshold([img, threshold], img)

    # Assert that the output matches the expected output
    assert_array_equal(output_data, expected_output)


def test_lower_threshold_otsu():
    # Create a sample nib.Nifti1Image object
    img_data = np.array([0, 1, 1, 50, 60, 60]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)

    # Otsu is expected to separate foreground and background
    expected_output = np.array([0, 0, 0, 1, 1, 1]).astype(float)

    output_data = lower_threshold_otsu([img], img)

    # compare output and expected arrays
    assert_array_equal(output_data, expected_output)


def test_upper_threshold_otsu():
    # Create a sample nib.Nifti1Image object
    img_data = np.array([0, 1, 1, 50, 60, 60]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)

    # Otsu is expected to separate foreground and background
    expected_output = np.array([1, 1, 1, 0, 0, 0]).astype(float)

    output_data = upper_threshold_otsu([img], img)

    # compare output and expected arrays
    assert_array_equal(output_data, expected_output)


def test_lower_clip():
    img_data = np.array([-1, 0, 1, 2, 3, 4]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)
    threshold = 1
    expected_output = np.array([1, 1, 1, 2, 3, 4])
    output_data = lower_clip([img, threshold], img)
    assert_array_equal(output_data, expected_output)


def test_upper_clip():
    img_data = np.array([-1, 0, 1, 2, 3, 4]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)
    threshold = 2
    expected_output = np.array([-1, 0, 1, 2, 2, 2])
    output_data = upper_clip([img, threshold], img)
    assert_array_equal(output_data, expected_output)


def test_absolute_value():
    img_data = np.array([-1, -2, 0, 1, 2]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)
    expected_output = np.array([1, 2, 0, 1, 2])
    output_data = absolute_value([img], img)
    assert_array_equal(output_data, expected_output)


def test_around():
    img_data = np.array([-1.1, -2.4, 0, 0.501, 1.1, 0.49]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)
    expected_output = np.array([-1, -2, 0, 1, 1, 0])
    output_data = around([img], img)
    assert_allclose(output_data, expected_output, rtol=EPSILON,
                    atol=EPSILON)


def test_ceil():
    img_data = np.array([-1.1, -0.5, 0, 0.5, 1.1]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)
    expected_output = np.array([-1, 0, 0, 1, 2])
    output_data = ceil([img], img)
    assert_array_equal(output_data, expected_output)


def test_floor():
    img_data = np.array([-1.1, -0.5, 0, 0.5, 1.1]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)
    expected_output = np.array([-2, -1, 0, 0, 1])
    output_data = floor([img], img)
    assert_array_equal(output_data, expected_output)


def test_normalize_sum():
    img_data = np.array([1, 2, 3, 4]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)
    expected_output = img_data / np.sum(img_data)
    output_data = normalize_sum([img], img)
    assert_array_almost_equal(output_data, expected_output)


def test_normalize_max():
    img_data = np.array([1, 2, 3, 4]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)
    expected_output = img_data / np.max(img_data)
    output_data = normalize_max([img], img)
    assert_array_almost_equal(output_data, expected_output)


def test_base_10_log():
    img_data = np.array([1, 10, 100, 1000]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)
    expected_output = np.log10(img_data)
    expected_output[0] = -65536
    output_data = base_10_log([img], img)
    assert_array_almost_equal(output_data, expected_output)


def test_natural_log():
    img_data = np.array([1, 2.71, 7.39, 20.09]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)
    expected_output = np.log(img_data)
    expected_output[0] = -65536
    output_data = natural_log([img], img)
    assert_array_almost_equal(output_data, expected_output)


def test_convert():
    img_data = np.array([1, 2, 3, 4], dtype=np.int16)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)
    expected_output = img_data.astype(np.float64)
    output_data = convert([img], img)
    assert_array_almost_equal(output_data, expected_output)


def test_addition():
    img_data_1 = np.array([1, 2]).astype(float)
    img_data_2 = np.array([3, 4]).astype(float)
    affine = np.eye(4)
    img1 = nib.Nifti1Image(img_data_1, affine)
    img2 = nib.Nifti1Image(img_data_2, affine)
    expected_output = img_data_1 + img_data_2
    output_data = addition([img1, img2], img1)
    assert_array_almost_equal(output_data, expected_output)


# Test function for subtraction
def test_subtraction():
    img_data_1 = np.array([1, 2]).astype(float)
    img_data_2 = np.array([3, 4]).astype(float)
    affine = np.eye(4)
    img1 = nib.Nifti1Image(img_data_1, affine)
    img2 = nib.Nifti1Image(img_data_2, affine)
    expected_output = img_data_1 - img_data_2
    output_data = subtraction([img1, img2], img1)
    assert_array_almost_equal(output_data, expected_output)


def test_multiplication():
    img_data_1 = np.array([1, 2]).astype(float)
    img_data_2 = np.array([3, 4]).astype(float)
    affine = np.eye(4)
    img1 = nib.Nifti1Image(img_data_1, affine)
    img2 = nib.Nifti1Image(img_data_2, affine)
    expected_output = img_data_1 * img_data_2
    output_data = multiplication([img1, img2], img1)
    assert_array_almost_equal(output_data, expected_output)


def test_division():
    img_data_1 = np.array([3, 4]).astype(float)
    img_data_2 = np.array([1, 2]).astype(float)
    affine = np.eye(4)
    img1 = nib.Nifti1Image(img_data_1, affine)
    img2 = nib.Nifti1Image(img_data_2, affine)
    expected_output = img_data_1 / img_data_2
    output_data = division([img1, img2], img1)
    assert_array_almost_equal(output_data, expected_output)


def test_mean():
    img_data_1 = np.array([1, 2]).astype(float)
    img_data_2 = np.array([3, 4]).astype(float)
    img_data_3 = np.array([5, 6]).astype(float)
    img_data_4 = np.array([7, 8]).astype(float)
    affine = np.eye(4)
    img1 = nib.Nifti1Image(img_data_1, affine)
    img2 = nib.Nifti1Image(img_data_2, affine)
    img3 = nib.Nifti1Image(img_data_3, affine)
    img4 = nib.Nifti1Image(img_data_4, affine)

    expected_output = (img_data_1 + img_data_2 + img_data_3 + img_data_4) / 4
    output_data = mean([img1, img2, img3, img4], img1)
    assert_array_almost_equal(output_data, expected_output)


def test_std():
    img_data_1 = np.array([1, 2]).astype(float)
    img_data_2 = np.array([3, 4]).astype(float)
    img_data_3 = np.array([5, 6]).astype(float)
    img_data_4 = np.array([7, 8]).astype(float)
    affine = np.eye(4)
    img1 = nib.Nifti1Image(img_data_1, affine)
    img2 = nib.Nifti1Image(img_data_2, affine)
    img3 = nib.Nifti1Image(img_data_3, affine)
    img4 = nib.Nifti1Image(img_data_4, affine)

    data_list = [img_data_1, img_data_2, img_data_3, img_data_4]
    mean_data = np.mean(data_list, axis=0)
    std_data = np.sqrt(np.sum((data_list - mean_data) ** 2, axis=0) / 4)

    output_data = std([img1, img2, img3, img4], img1)
    assert_array_almost_equal(output_data, std_data)


def test_union():
    img_data_1 = np.array([0, 1]).astype(float)
    img_data_2 = np.array([1, 0]).astype(float)
    affine = np.eye(4)
    img1 = nib.Nifti1Image(img_data_1, affine)
    img2 = nib.Nifti1Image(img_data_2, affine)

    expected_output = np.array([1, 1])
    output_data = union([img1, img2], img1)
    assert_array_almost_equal(output_data, expected_output)


def test_intersection():
    img_data_1 = np.array([1, 1]).astype(float)
    img_data_2 = np.array([1, 0]).astype(float)
    affine = np.eye(4)
    img1 = nib.Nifti1Image(img_data_1, affine)
    img2 = nib.Nifti1Image(img_data_2, affine)

    expected_output = np.array([1, 0])
    output_data = intersection([img1, img2], img1)
    assert_array_almost_equal(output_data, expected_output)


def test_difference():
    img_data_1 = np.array([1, 1]).astype(float)
    img_data_2 = np.array([1, 0]).astype(float)
    affine = np.eye(4)
    img1 = nib.Nifti1Image(img_data_1, affine)
    img2 = nib.Nifti1Image(img_data_2, affine)

    expected_output = np.array([0, 1])
    output_data = difference([img1, img2], img1)
    assert_array_almost_equal(output_data, expected_output)


def test_invert():
    img_data = np.array([1, 0]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)

    expected_output = np.array([0, 1])
    output_data = invert([img], img)
    assert_array_almost_equal(output_data, expected_output)


def test_concatenate():
    img_data_1 = np.zeros((3, 3, 3)).astype(float)
    img_data_2 = np.ones((3, 3, 3)).astype(float)
    affine = np.eye(4)
    img1 = nib.Nifti1Image(img_data_1, affine)
    img2 = nib.Nifti1Image(img_data_2, affine)

    output_data = concatenate([img1, img2], img1)
    assert_array_almost_equal(img_data_1.shape+(2,), output_data.shape)


def test_get_neighbors():
    # Input data: small data with NOT the same dimension in each direction.
    data = np.arange(3 * 4 * 5).reshape((3, 4, 5)) + 1

    # Cutting into patches of radius 1, i.e. 3x3x3.
    result = _get_neighbors(data, radius=1)

    # Expected output shape: Should fit 3 x 4 x 5 patches of shape 3x3x3
    expected_shape = (3, 4, 5, 3, 3, 3)
    assert result.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {result.shape}"

    # First block: expecting 8 non-zero values (always true with patch_size 3!)
    # (draw a cube and check the number of neighbors of the voxel at the
    #  corner)
    assert np.count_nonzero(result[0, 0, 0, :, :, :]) == 8

    # Middle blocks: expecting 27 non-zero values
    assert np.count_nonzero(result[1, 1, 1, :, :, :]) == 27

    # Comparing with values obtained the day we created this test.
    # Expected first block:
    expected_first_block = np.array([[[0,  0,  0],
                                      [0,  0,  0],
                                      [0,  0,  0]],
                                     [[0,  0,  0],
                                      [0,  1,  2],
                                      [0,  6,  7]],
                                     [[0,  0,  0],
                                      [0, 21, 22],
                                      [0, 26, 27]]])
    assert_array_equal(result[0, 0, 0, :, :, :], expected_first_block)

    # Expected last block
    expected_last_block = np.array([[[34, 35,  0],
                                     [39, 40,  0],
                                     [0,   0,  0]],
                                    [[54, 55,  0],
                                     [59, 60,  0],
                                     [0,  0,  0]],
                                    [[0,  0,  0],
                                     [0,  0,  0],
                                     [0,  0,  0]]])
    assert_array_equal(result[-1, -1, -1, :, :, :], expected_last_block)


def test_neighborhood_correlation():
    # Note. Not working on 2D data.
    affine = np.eye(4)

    # Test 1: Perfect correlation
    # Compares uniform patch of ones with a uniform patch of twos.
    # No background.
    img_data_1 = np.ones((3, 3, 3), dtype=float)
    img1 = nib.Nifti1Image(img_data_1, affine)

    img_data_2 = np.ones((3, 3, 3), dtype=float) * 2
    img2 = nib.Nifti1Image(img_data_2, affine)
    output = neighborhood_correlation([img1, img2], img1)
    assert np.allclose(output, np.ones((3, 3, 3))), \
        "Expected a perfect correlation, got: {}".format(output)

    # Test 2: Bad correlation.
    # Compares uniform patch of ones with a noisy patch of twos.
    # There should be a poor correlation (cloud of points is horizontal).
    # But we notice that around the edges of the image, high correlation (~1),
    # as explained in the correlation method's docstring
    img_data_2 = np.ones((3, 3, 3), dtype=float) * 2 + \
        np.random.rand(3, 3, 3) * 0.001
    img2 = nib.Nifti1Image(img_data_2, affine)
    output = neighborhood_correlation([img1, img2], img1)
    expected = np.ones((3, 3, 3))
    expected[1, 1, 1] = 0
    assert np.allclose(output, expected), \
        ("Expected a bad correlation at central point, good around the border,"
         " got: {}").format(output)

    # Test 2: Comparing with only background: should be 0 everywhere.
    img_data_2 = np.zeros((3, 3, 3)).astype(float)
    img2 = nib.Nifti1Image(img_data_2, affine)
    output = neighborhood_correlation([img1, img2], img1)
    assert np.allclose(output, np.zeros((3, 3, 3))), \
        "Expected a 0 correlation everywhere, got {}".format(output)


def test_dilation():
    img_data = np.array([0, 1]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)

    expected_output = binary_dilation(img_data, iterations=1)
    output_data = dilation([img, 1], img)
    assert_array_almost_equal(output_data, expected_output)


def test_erosion():
    img_data = np.array([0, 1]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)

    expected_output = binary_erosion(img_data, iterations=1)
    output_data = erosion([img, 1], img)
    assert_array_almost_equal(output_data, expected_output)


def test_closing():
    img_data = np.array([0, 1]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)

    expected_output = binary_closing(img_data, iterations=1)
    output_data = closing([img, 1], img)
    assert_array_almost_equal(output_data, expected_output)


def test_opening():
    img_data = np.array([0, 1]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)

    expected_output = binary_opening(img_data, iterations=1)
    output_data = opening([img, 1], img)
    assert_array_almost_equal(output_data, expected_output)


def test_gaussian_blur():
    img_data = np.array([0, 1]).astype(float)
    affine = np.eye(4)
    img = nib.Nifti1Image(img_data, affine)

    expected_output = gaussian_filter(img_data, sigma=1)
    output_data = gaussian_blur([img, 1], img)
    assert_array_almost_equal(output_data, expected_output)
