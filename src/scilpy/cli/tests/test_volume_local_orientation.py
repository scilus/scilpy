#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()
in_img = os.path.join(SCILPY_HOME, 'others', 'fa.nii.gz')


def test_help_option(script_runner):
    ret = script_runner.run(['scil_volume_local_orientation', '--help'])
    assert ret.success


def test_structure_tensor(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    out_direction = 'direction.nii.gz'
    out_probability = 'probability.nii.gz'
    ret = script_runner.run([
        'scil_volume_local_orientation',
        in_img, out_direction, out_probability,
        '--method', 'structure_tensor'])

    assert ret.success

    in_data = nib.load(in_img)
    direction = nib.load(out_direction)
    probability = nib.load(out_probability)

    assert direction.shape == in_data.shape + (3,)
    assert probability.shape == in_data.shape
    assert np.allclose(direction.affine, in_data.affine)
    assert np.allclose(probability.affine, in_data.affine)
    assert np.isfinite(direction.get_fdata()).all()
    assert np.isfinite(probability.get_fdata()).all()


def test_frangi(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    out_direction = 'direction_frangi.nii.gz'
    out_probability = 'probability_frangi.nii.gz'
    ret = script_runner.run([
        'scil_volume_local_orientation',
        in_img, out_direction, out_probability])

    assert ret.success

    in_data = nib.load(in_img)
    direction = nib.load(out_direction)
    probability = nib.load(out_probability)

    assert direction.shape == in_data.shape + (3,)
    assert probability.shape == in_data.shape
    assert np.allclose(direction.affine, in_data.affine)
    assert np.allclose(probability.affine, in_data.affine)
    assert np.isfinite(direction.get_fdata()).all()
    assert np.isfinite(probability.get_fdata()).all()


def test_frangi_multiscale(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    out_direction = 'direction_frangi.nii.gz'
    out_probability = 'probability_frangi.nii.gz'
    ret = script_runner.run([
        'scil_volume_local_orientation',
        in_img, out_direction, out_probability,
        '--sigma', '1.0', '2.0'])

    assert ret.success

    in_data = nib.load(in_img)
    direction = nib.load(out_direction)
    probability = nib.load(out_probability)

    assert direction.shape == in_data.shape + (3,)
    assert probability.shape == in_data.shape
    assert np.allclose(direction.affine, in_data.affine)
    assert np.allclose(probability.affine, in_data.affine)
    assert np.isfinite(direction.get_fdata()).all()
    assert np.isfinite(probability.get_fdata()).all()

