import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scilpy.utils.spatial import split_affine_transform


def _compose_affine(translation, rotation, shear, scale):
    affine = np.eye(4)
    affine[:3, 3] = translation
    affine[:3, :3] = rotation @ shear @ scale
    return affine


def test_split_affine_transform_recovers_components():
    translation = np.array([1.2, -3.4, 5.6])
    rotation = Rotation.from_euler('XYZ', [0.3, -0.2, 0.4]).as_matrix()
    shear = np.array([[1., 0.2, -0.4],
                      [0., 1., 0.3],
                      [0., 0., 1.]])
    scale = np.diag([2.0, 3.0, 4.0])

    affine = _compose_affine(translation, rotation, shear, scale)
    split_translation, split_rotation, split_scale, split_shear = \
        split_affine_transform(affine)

    np.testing.assert_allclose(split_translation[:3, 3], translation)
    np.testing.assert_allclose(split_rotation[:3, :3], rotation)
    np.testing.assert_allclose(split_scale[:3, :3], scale)
    np.testing.assert_allclose(split_shear[:3, :3], shear)

    recomposed = split_translation @ split_rotation @ split_shear @ split_scale
    np.testing.assert_allclose(recomposed, affine)


def test_split_affine_transform_keeps_rotation_proper():
    affine = np.diag([-2.0, 3.0, 4.0, 1.0])

    _, rotation, scale, shear = split_affine_transform(affine)

    np.testing.assert_allclose(rotation[:3, :3].T @ rotation[:3, :3],
                               np.eye(3))
    assert np.isclose(np.linalg.det(rotation[:3, :3]), 1.0)
    assert np.any(np.diag(scale[:3, :3]) < 0)
    np.testing.assert_allclose(shear, np.eye(4))
    np.testing.assert_allclose(rotation @ shear @ scale, affine)


def test_split_affine_transform_rejects_degenerate_scale():
    affine = np.diag([1.0, 0.0, 2.0, 1.0])

    with pytest.raises(ValueError, match='singular or near-singular'):
        split_affine_transform(affine)
