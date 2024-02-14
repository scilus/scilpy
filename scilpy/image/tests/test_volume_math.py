# -*- coding: utf-8 -*-


import nibabel as nib
import numpy as np
from numpy.testing import (assert_array_equal,
                           assert_allclose,
                           assert_array_almost_equal)
from scipy.ndimage import (binary_closing, binary_dilation,
                           binary_erosion, binary_opening,
                           gaussian_filter)


from scilpy.image.volume_math import (_validate_imgs,
                                      _validate_imgs_concat,
                                      _validate_length,
                                      _validate_type,
                                      _validate_float,
                                      cut_up_cube,
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
                                      closing, opening)


EPSILON = np.finfo(float).eps


def test_validate_imgs_matching():
    data1 = np.zeros((5, 5, 5)).astype(float)
    affine1 = np.eye(4)
    img1 = nib.Nifti1Image(data1, affine1)

    data2 = np.zeros((5, 5, 5)).astype(float)
    affine2 = np.eye(4)
    img2 = nib.Nifti1Image(data2, affine2)

    try:
        _validate_imgs(img1, img2)
        print("Test with matching shapes passed.")
    except ValueError as e:
        print("Test with matching shapes failed:", str(e))


def test_validate_imgs_different():
    data1 = np.zeros((5, 5, 5)).astype(float)
    affine1 = np.eye(4)
    img1 = nib.Nifti1Image(data1, affine1)

    data2 = np.zeros((6, 6, 6)).astype(float)
    affine2 = np.eye(4)
    img2 = nib.Nifti1Image(data2, affine2)

    try:
        _validate_imgs(img1, img2)
        print("Test with different shapes passed.")
    except ValueError as e:
        print("Test with different shapes failed:", str(e))


def test_validate_imgs_concat_all():
    data1 = np.zeros((5, 5, 5)).astype(float)
    affine1 = np.eye(4)
    img1 = nib.Nifti1Image(data1, affine1)

    data2 = np.zeros((6, 6, 6)).astype(float)
    affine2 = np.eye(4)
    img2 = nib.Nifti1Image(data2, affine2)

    try:
        _validate_imgs_concat(img1, img2)
        print("Test with all valid NIFTI images passed.")
    except ValueError as e:
        print("Test with all valid NIFTI images failed:", str(e))


def test_validate_imgs_concat_not_all():
    data1 = np.zeros((5, 5, 5)).astype(float)
    affine1 = np.eye(4)
    img1 = nib.Nifti1Image(data1, affine1)

    not_an_image = "I am not an image"

    try:
        _validate_imgs_concat(img1, not_an_image)
        print("Test with one invalid object passed.")
    except ValueError as e:
        print("Test with one invalid object failed:", str(e))


def test_validate_length_exact():
    exact_length_list = [1, 2, 3]
    shorter_list = [1, 2]

    try:
        _validate_length(exact_length_list, 3)
        print("Test with exact length passed.")
    except ValueError as e:
        print("Test with exact length failed:", str(e))

    try:
        _validate_length(shorter_list, 3)
        print("Test with shorter length passed.")
    except ValueError as e:
        print("Test with shorter length failed:", str(e))


def test_validate_length_at_least():
    at_least_list = [1, 2, 3, 4]
    shorter_list = [1, 2]

    try:
        _validate_length(at_least_list, 3, at_least=True)
        print("Test with 'at least' length passed.")
    except ValueError as e:
        print("Test with 'at least' length failed:", str(e))

    try:
        _validate_length(shorter_list, 3, at_least=True)
        print("Test with shorter 'at least' length passed.")
    except ValueError as e:
        print("Test with shorter 'at least' length failed:", str(e))


def test_validate_type_correct():
    correct_type_input = 42  # Integer

    try:
        _validate_type(correct_type_input, int)
        print("Test with correct type passed.")
    except ValueError:
        print("Test with correct type failed.")


def test_validate_type_incorrect():
    incorrect_type_input = "42"  # String

    try:
        _validate_type(incorrect_type_input, int)
        print("Test with incorrect type passed.")
    except ValueError:
        print("Test with incorrect type failed.")


def test_validate_float_correct():
    correct_input = 42  # Integer can be cast to float

    try:
        _validate_float(correct_input)
        print("Test with correct type passed.")
    except ValueError:
        print("Test with correct type failed.")


def test_validate_float_incorrect():
    incorrect_input = "not_a_float"  # String that can't be cast to float

    try:
        _validate_float(incorrect_input)
        print("Test with incorrect type passed.")
    except ValueError:
        print("Test with incorrect type failed.")


def test_cut_up_cube_with_known_output():
    # Input data: smaller 3x3x3 cube
    data = np.arange(3 * 3 * 3).reshape((3, 3, 3))
    blck = (3, 3, 3)

    # Running the function
    result = cut_up_cube(data, blck)

    # Expected output shape
    expected_shape = (3, 3, 3, 3, 3, 3)

    # Expected first block
    expected_first_block = np.array([[[0,  0,  0],
                                      [0,  0,  0],
                                      [0,  0,  0]],

                                    [[0,  0,  0],
                                     [0,  0,  1],
                                     [0,  3,  4]],

                                    [[0,  0,  0],
                                     [0,  9, 10],
                                     [0, 12, 13]]])

    # Expected last block
    expected_last_block = np.array([[[13, 14,  0],
                                     [16, 17,  0],
                                     [0,  0,  0]],

                                   [[22, 23,  0],
                                    [25, 26,  0],
                                    [0,  0,  0]],

                                   [[0,  0,  0],
                                    [0,  0,  0],
                                    [0,  0,  0]]])

    # Asserting that the output shape matches the expected shape
    assert result.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {result.shape}"

    # Asserting that the first block matches the expected first block
    assert_array_equal(result[0, 0, 0, :, :, :], expected_first_block)

    # Asserting that the last block matches the expected last block
    assert_array_equal(result[-1, -1, -1, :, :, :], expected_last_block)


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


# Test function for upper_threshold_eq
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
