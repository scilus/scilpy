#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import numpy as np
import nibabel as nib

from scilpy.io.streamlines import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin

tmp_dir = tempfile.TemporaryDirectory()


def init_data():
    streamlines = [[[5., 1., 5.], [5., 5., 9.], [7., 9., 9.], [13., 11., 9.],
                    [5., 7., 7.]], [[7., 7., 7.], [9., 9., 9.]]]

    mask = np.ones((15, 15, 15))
    affine = np.eye(4)
    header = nib.nifti2.Nifti2Header()
    extra = {
        'affine': affine,
        'dimensions': (15, 15, 15),
        'voxel_size': 1.,
        'voxel_order': "RAS"
    }
    mask_img = nib.nifti2.Nifti2Image(mask, affine, header, extra)

    sft = StatefulTractogram(streamlines, mask_img,
                             space=Space.VOX, origin=Origin.NIFTI)
    save_tractogram(sft, 'tractogram.trk', True)


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_filter_collisions.py', '--help')
    assert ret.success


def test_execution_filtering(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()

    diameters = [5, 1]
    np.savetxt('diameters.txt', diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            'tractogram.trk', 'diameters.txt', 'clean.trk',
                            '-f')
    assert ret.success


def test_execution_filtering_out_colliding_prefix(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()

    diameters = [5, 1]
    np.savetxt('diameters.txt', diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            'tractogram.trk', 'diameters.txt', 'clean.trk',
                            '--out_colliding_prefix', 'tractogram', '-f')
    assert ret.success


def test_execution_filtering_single_diameter(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()

    diameters = [5]
    np.savetxt('diameters.txt', diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            'tractogram.trk', 'diameters.txt', 'clean.trk',
                            '-f')
    assert ret.success


def test_execution_filtering_no_shuffle(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()

    diameters = [5, 1]
    np.savetxt('diameters.txt', diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            'tractogram.trk', 'diameters.txt', 'clean.trk',
                            '--disable_shuffling', '-f')
    assert ret.success


def test_execution_filtering_min_distance(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()

    diameters = [0.001, 0.001]
    np.savetxt('diameters.txt', diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            'tractogram.trk', 'diameters.txt', 'clean.trk',
                            '--min_distance', '5', '-f')
    assert ret.success


def test_execution_filtering_metrics(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()

    # No collision, as we want to keep two streamlines for this test.
    diameters = [0.001, 0.001]
    np.savetxt('diameters.txt', diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            'tractogram.trk', 'diameters.txt', 'clean.trk',
                            '--out_metrics', 'metrics.json', '-f')
    assert ret.success


def test_execution_rotation_matrix(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()

    # No collision, as we want to keep two streamlines for this test.
    diameters = [0.001, 0.001]
    np.savetxt('diameters.txt', diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            'tractogram.trk', 'diameters.txt', 'clean.trk',
                            '--out_rotation_matrix', 'rotation.mat', '-f')
    assert ret.success
