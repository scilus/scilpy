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
                             space=Space.VOX,
                             origin=Origin.NIFTI)
    sft.data_per_streamline = {
        "diameters": [0.002, 0.001]
    }

    save_tractogram(sft, 'tractogram.trk', True)


def test_help_option(script_runner):
    ret = script_runner.run('scil_fibertube_tracking.py', '--help')
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_tracking.py',
                            'tractogram.trk', 'tracking.trk',
                            '--min_length', '0', '-f')
    assert ret.success


def test_execution_rk(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_tracking.py',
                            'tractogram.trk', 'tracking.trk',
                            '--rk_order', '2', '--min_length', '0', '-f')
    assert ret.success


def test_execution_config(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_tracking.py',
                            'tractogram.trk', 'tracking.trk',
                            '--blur_radius', '0.3',
                            '--step_size', '0.1',
                            '--out_config', 'config.json',
                            '--min_length', '0', '-f')
    assert ret.success


def test_execution_seeding(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_tracking.py',
                            'tractogram.trk', 'tracking.trk',
                            '--nb_fibertubes', '1',
                            '--nb_seeds_per_fibertube', '3', '--skip', '3',
                            '--local_seeding', 'center',
                            '--min_length', '0', '-f')
    assert ret.success


def test_execution_FTODF(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_tracking.py',
                            'tractogram.trk', 'tracking.trk', '--use_ftODF',
                            '--min_length', '0', '-f')
    assert ret.success


def test_execution_FTODF_rk(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_tracking.py',
                            'tractogram.trk', 'tracking.trk', '--use_ftODF',
                            '--rk_order', '2', '--min_length', '0', '-f')
    assert ret.success


def test_execution_FTODF_config(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_tracking.py',
                            'tractogram.trk', 'tracking.trk', '--use_ftODF',
                            '--blur_radius', '0.3',
                            '--step_size', '0.1',
                            '--out_config', 'config.json',
                            '--min_length', '0', '-f')
    assert ret.success


def test_execution_FTODF_seeding(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_tracking.py',
                            'tractogram.trk', 'tracking.trk', '--use_ftODF',
                            '--nb_fibertubes', '1',
                            '--nb_seeds_per_fibertube', '3', '--skip', '3',
                            '--local_seeding', 'center',
                            '--min_length', '0', '-f')
    assert ret.success


def test_execution_FTODF_sphere(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_tracking.py',
                            'tractogram.trk', 'tracking.trk', '--use_ftODF',
                            '--sh_order', '4',
                            '--sphere', 'symmetric362',
                            '--sh_basis', 'tournier07',
                            '--sub_sphere', '0',
                            '--sfthres', '0.05',
                            '--sfthres_init', '0.4',
                            '--min_length', '0', '-f')
    assert ret.success


def test_execution_FTODF_det(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_tracking.py',
                            'tractogram.trk', 'tracking.trk', '--use_ftODF',
                            '--algo', 'det',
                            '--min_length', '0', '-f')
    assert ret.success
