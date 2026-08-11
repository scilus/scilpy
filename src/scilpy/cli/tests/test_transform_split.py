#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import numpy as np
from scipy.io import savemat
from scipy.spatial.transform import Rotation

tmp_dir = tempfile.TemporaryDirectory()


def _write_numeric_affine(path, translation, rotation, shear, scale):
    affine = np.eye(4)
    affine[:3, 3] = translation
    affine[:3, :3] = rotation @ shear @ scale
    np.savetxt(path, affine)
    return affine


def _write_itk_affine(path, rotation_ras, translation_ras):
    lps2ras = np.diag([-1., -1., 1.])
    rotation_lps = lps2ras @ rotation_ras @ lps2ras
    translation_lps = np.array([-translation_ras[0],
                                -translation_ras[1],
                                translation_ras[2]])

    parameters = np.concatenate((rotation_lps.reshape(-1), translation_lps))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('#Insight Transform File V1.0\n')
        f.write('# Transform 0\n')
        f.write('Transform: AffineTransform_double_3_3\n')
        f.write('Parameters: {}\n'.format(
            ' '.join('{:.18e}'.format(v) for v in parameters)))
        f.write('FixedParameters: 0 0 0\n')


def test_help_option(script_runner):
    ret = script_runner.run(['scil_transform_split', '--help'])
    assert ret.success


def test_fails_without_output(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    np.savetxt('affine.txt', np.eye(4))

    ret = script_runner.run(['scil_transform_split', 'affine.txt'])
    assert not ret.success


def test_execution_itk_translation(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    translation = np.array([1.5, -2.5, 3.5])
    _write_itk_affine('affine_itk.txt', np.eye(3), translation)

    ret = script_runner.run(['scil_transform_split', 'affine_itk.txt',
                             '--translation', 'translation.txt', '-f'])
    assert ret.success

    out_translation = np.loadtxt('translation.txt')
    expected = np.eye(4)
    expected[:3, 3] = translation
    np.testing.assert_allclose(out_translation, expected)


def test_execution_ants_mat_translation(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    translation = np.array([1.5, -2.5, 3.5])
    parameters = np.concatenate((np.eye(3).reshape(-1),
                                 [-translation[0], -translation[1],
                                  translation[2]]))
    savemat('affine.mat', {
        'AffineTransform_double_3_3': parameters,
        'fixed': np.zeros(3)
    })

    ret = script_runner.run(['scil_transform_split', 'affine.mat',
                             '--translation', 'translation.txt', '-f'])
    assert ret.success

    out_translation = np.loadtxt('translation.txt')
    expected = np.eye(4)
    expected[:3, 3] = translation
    np.testing.assert_allclose(out_translation, expected)


def test_execution_all_matrix_outputs(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    translation = np.array([1.2, -3.4, 5.6])
    rotation = Rotation.from_euler('XYZ', [0.3, -0.2, 0.4]).as_matrix()
    shear = np.array([[1., 0.2, -0.4],
                      [0., 1., 0.3],
                      [0., 0., 1.]])
    scale = np.diag([2.0, 3.0, 4.0])
    affine = _write_numeric_affine('affine.txt', translation, rotation,
                                   shear, scale)

    ret = script_runner.run([
        'scil_transform_split', 'affine.txt',
        '--translation', 'translation.txt',
        '--rotation', 'rotation.txt',
        '--scale', 'scale.txt',
        '--shear', 'shear.txt',
        '-f'
    ])
    assert ret.success

    out_translation = np.loadtxt('translation.txt')
    out_rotation = np.loadtxt('rotation.txt')
    out_scale = np.loadtxt('scale.txt')
    out_shear = np.loadtxt('shear.txt')

    np.testing.assert_allclose(out_translation[:3, 3], translation)
    np.testing.assert_allclose(out_rotation[:3, :3], rotation)
    np.testing.assert_allclose(out_scale[:3, :3], scale)
    np.testing.assert_allclose(out_shear[:3, :3], shear)
    np.testing.assert_allclose(
        out_translation @ out_rotation @ out_shear @ out_scale,
        affine)


def test_angles_output(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    angles = np.array([0.3, -0.2, 0.4])
    rotation = Rotation.from_euler('XYZ', angles).as_matrix()
    _write_numeric_affine('affine.txt', np.zeros(3), rotation, np.eye(3),
                          np.eye(3))

    ret = script_runner.run(['scil_transform_split', 'affine.txt',
                             '--rotation', 'rotation.txt',
                             '--angles', '-f'])
    assert ret.success

    out_angles = np.loadtxt('rotation.txt')
    np.testing.assert_allclose(out_angles, angles)


def test_rodrigues_output(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    rotation = Rotation.from_euler('XYZ', [0.3, -0.2, 0.4]).as_matrix()
    _write_numeric_affine('affine.txt', np.zeros(3), rotation, np.eye(3),
                          np.eye(3))

    ret = script_runner.run(['scil_transform_split', 'affine.txt',
                             '--rotation', 'rotation.txt',
                             '--rodrigues', '-f'])
    assert ret.success

    out_rodrigues = np.loadtxt('rotation.txt')
    expected_rotvec = Rotation.from_matrix(rotation).as_rotvec()
    expected_angle = np.linalg.norm(expected_rotvec)
    expected_axis = expected_rotvec / expected_angle
    expected = np.concatenate(([expected_angle], expected_axis))
    np.testing.assert_allclose(out_rodrigues, expected)


def test_rotation_encoding_requires_rotation(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    np.savetxt('affine.txt', np.eye(4))

    ret = script_runner.run(['scil_transform_split', 'affine.txt',
                             '--angles'])
    assert not ret.success

    ret = script_runner.run(['scil_transform_split', 'affine.txt',
                             '--rodrigues'])
    assert not ret.success


def test_rotation_output_modes_are_mutually_exclusive(script_runner,
                                                      monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    np.savetxt('affine.txt', np.eye(4))

    ret = script_runner.run(['scil_transform_split', 'affine.txt',
                             '--rotation', 'rotation.txt',
                             '--angles', '--rodrigues'])
    assert not ret.success


def test_rejects_composite_itk_transform(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    with open('affine_itk.txt', 'w', encoding='utf-8') as f:
        f.write('#Insight Transform File V1.0\n')
        f.write('# Transform 0\n')
        f.write('Transform: AffineTransform_double_3_3\n')
        f.write('Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\n')
        f.write('FixedParameters: 0 0 0\n')
        f.write('# Transform 1\n')
        f.write('Transform: AffineTransform_double_3_3\n')
        f.write('Parameters: 1 0 0 0 1 0 0 0 1 1 2 3\n')
        f.write('FixedParameters: 0 0 0\n')

    ret = script_runner.run(['scil_transform_split', 'affine_itk.txt',
                             '--translation', 'translation.txt', '-f'])
    assert not ret.success


def test_rejects_singular_affine(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    np.savetxt('affine.txt', np.diag([1., 0., 2., 1.]))

    ret = script_runner.run(['scil_transform_split', 'affine.txt',
                             '--translation', 'translation.txt', '-f'])
    assert not ret.success


def test_rejects_malformed_itk_transform(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    with open('affine_itk.txt', 'w', encoding='utf-8') as f:
        f.write('#Insight Transform File V1.0\n')
        f.write('Transform: AffineTransform_double_3_3\n')

    ret = script_runner.run(['scil_transform_split', 'affine_itk.txt',
                             '--translation', 'translation.txt', '-f'])
    assert not ret.success
