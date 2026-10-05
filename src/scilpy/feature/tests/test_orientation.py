import numpy as np

from scilpy.feature.orientation import (
    compute_background_score,
    compute_blob_like_score,
    compute_frangi_features,
    compute_plate_like_score,
    compute_scaled_orientation,
    compute_structureness,
    divide_nonzero,
    frangi_filter,
    reject_vesselness_background,
    sort_eigen,
)


def _make_test_volume(shape=(5, 5, 5)):
    grid = np.indices(shape, dtype=np.float32)
    center = np.array([(dim - 1) / 2 for dim in shape], dtype=np.float32)
    squared_distance = np.sum(
        (grid - center[:, None, None, None]) ** 2, axis=0)
    return np.exp(-squared_distance / 4.0).astype(np.float32)


def test_divide_nonzero_handles_zero_denominator():
    numerator = np.array([2.0, -4.0, 0.0], dtype=np.float32)
    denominator = np.array([1.0, 0.0, 2.0], dtype=np.float32)

    result = divide_nonzero(numerator, denominator, new_val=0.5)

    np.testing.assert_allclose(result, np.array(
        [2.0, -8.0, 0.0], dtype=np.float32))
    assert np.isfinite(result).all()


def test_sort_eigen_sorts_by_absolute_value_and_reorders_vectors():
    eigval = np.array([[[[3.0, -1.0, 2.0]]]], dtype=np.float32)
    eigvec = np.eye(3, dtype=np.float32)[None, None, None, :, :]

    srt_eigval, srt_eigvec = sort_eigen(eigval, eigvec)

    np.testing.assert_allclose(
        np.array([value.item() for value in srt_eigval], dtype=np.float32),
        np.array([-1.0, 2.0, 3.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(srt_eigvec[0, 0, 0, :, 0], np.array([
                                  0.0, 1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(srt_eigvec[0, 0, 0, :, 1], np.array([
                                  0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(srt_eigvec[0, 0, 0, :, 2], np.array([
                                  1.0, 0.0, 0.0], dtype=np.float32))


def test_scalar_frangi_score_helpers_match_formulas():
    ra = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    rb = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    s = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    np.testing.assert_allclose(
        compute_plate_like_score(ra, alpha=1.0),
        1 - np.exp(-(ra ** 2) / 2.0),
    )
    np.testing.assert_allclose(
        compute_blob_like_score(rb, beta=1.0),
        np.exp(-(rb ** 2) / 2.0),
    )
    np.testing.assert_allclose(
        compute_background_score(s, gamma=1.0),
        1 - np.exp(-(s ** 2) / 2.0),
    )
    np.testing.assert_allclose(
        compute_structureness(ra, rb, s),
        np.sqrt(ra ** 2 + rb ** 2 + s ** 2),
    )


def test_compute_frangi_features_returns_expected_scores_and_default_gamma():
    eigen1 = np.array([[[-2.0]]], dtype=np.float32)
    eigen2 = np.array([[[3.0]]], dtype=np.float32)
    eigen3 = np.array([[[-6.0]]], dtype=np.float32)

    ra, rb, s, gamma = compute_frangi_features(
        eigen1, eigen2, eigen3, gamma=None)

    np.testing.assert_allclose(ra, np.array([[[0.5]]], dtype=np.float32))
    np.testing.assert_allclose(rb, np.array(
        [[[2.0 / np.sqrt(18.0)]]], dtype=np.float32))
    np.testing.assert_allclose(s, np.array([[[7.0]]], dtype=np.float32))
    assert gamma == 3.5


def test_reject_vesselness_background_masks_nan_and_positive_secondary_eigenvalues():
    vesselness = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float32)
    eigen2 = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    eigen3 = np.array([-1.0, -1.0, 1.0, -1.0], dtype=np.float32)

    masked = reject_vesselness_background(vesselness.copy(), eigen2, eigen3)

    np.testing.assert_allclose(masked, np.array(
        [1.0, 0.0, 0.0, 0.0], dtype=np.float32), equal_nan=True)


def test_compute_scaled_orientation_returns_finite_arrays():
    img = _make_test_volume()

    frangi_img, eigvec, eigval = compute_scaled_orientation(
        1,
        img,
        alpha=0.001,
        beta=1.0,
        gamma=None,
    )

    assert frangi_img.shape == img.shape
    assert eigvec.shape == img.shape + (3,)
    assert eigval.shape == img.shape + (3,)
    assert np.isfinite(frangi_img).all()
    assert np.isfinite(eigvec).all()
    assert np.isfinite(eigval).all()
    assert np.all(frangi_img >= 0)


def test_frangi_filter_matches_single_scale_orientation_smoke():
    img = _make_test_volume()

    frangi_img, eigvec, _ = compute_scaled_orientation(
        1,
        img,
        alpha=0.001,
        beta=1.0,
        gamma=None,
    )
    filtered_frangi, filtered_vec = frangi_filter(
        img,
        scales_px=[1],
        alpha=0.001,
        beta=1.0,
        gamma=None,
    )

    np.testing.assert_allclose(filtered_frangi, frangi_img)
    np.testing.assert_allclose(filtered_vec, eigvec)
