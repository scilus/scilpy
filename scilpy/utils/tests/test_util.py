# -*- coding: utf-8 -*-
import numpy as np
from numpy.testing import assert_almost_equal

from scilpy.utils.util import rotation_around_vector_matrix


def test_output_shape_and_type():
    """Test the output shape and type."""
    vec = np.array([1, 0, 0])
    theta = np.pi / 4  # 45 degrees
    rot_matrix = rotation_around_vector_matrix(vec, theta)
    assert isinstance(rot_matrix, np.ndarray)
    assert np.array_equal(rot_matrix.shape, (3, 3))


def test_magnitude_preservation():
    """Test if the rotation preserves the magnitude of a vector."""
    vec = np.array([1, 0, 0])
    theta = np.pi / 4
    rot_matrix = rotation_around_vector_matrix(vec, theta)
    rotated_vec = np.dot(rot_matrix, vec)
    assert_almost_equal(np.linalg.norm(rotated_vec), np.linalg.norm(vec),
                        decimal=5)


def test_known_rotation():
    """Test a known rotation case."""
    vec = np.array([0, 0, 1])  # Rotation around z-axis
    theta = np.pi / 2  # 90 degrees
    rot_matrix = rotation_around_vector_matrix(vec, theta)
    original_vec = np.array([1, 0, 0])
    expected_rotated_vec = np.array([0, 1, 0])
    rotated_vec = np.dot(rot_matrix, original_vec)
    assert_almost_equal(rotated_vec, expected_rotated_vec, decimal=5)


def test_zero_rotation():
    """Test rotation with theta = 0."""
    vec = np.array([1, 0, 0])
    theta = 0
    rot_matrix = rotation_around_vector_matrix(vec, theta)
    np.array_equal(rot_matrix, np.eye(3))


def test_full_rotation():
    """Test rotation with theta = 2*pi (should be identity)."""
    vec = np.array([1, 0, 0])
    theta = 2 * np.pi
    rot_matrix = rotation_around_vector_matrix(vec, theta)
    # Allow for minor floating-point errors
    assert_almost_equal(rot_matrix, np.eye(3), decimal=5)
