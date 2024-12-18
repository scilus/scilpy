#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
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
    header = nib.Nifti1Header()
    extra = {
        'affine': affine,
        'dimensions': (15, 15, 15),
        'voxel_size': 1.,
        'voxel_order': "RAS"
    }
    mask_img = nib.Nifti1Image(mask, affine, header, extra)

    sft_fibertubes = StatefulTractogram(streamlines, mask_img, Space.VOX,
                                        Origin.NIFTI)
    sft_fibertubes.data_per_streamline = {
        "diameters": [0.2, 0.01]
    }

    save_tractogram(sft_fibertubes, 'fibertubes.trk', True)


def test_help_option(script_runner):
    ret = script_runner.run('scil_fibertube_compute_density.py', '--help')
    assert ret.success


def test_execution_density(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_compute_density.py',
                            'fibertubes.trk',
                            '--out_density_map', 'density_map.nii.gz',
                            '--out_density_measures',
                            'density_measures.json',
                            '-f')
    assert ret.success


def test_execution_collisions(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_compute_density.py',
                            'fibertubes.trk',
                            '--out_collision_map', 'collision_map.nii.gz',
                            '--out_collision_measures',
                            'collision_measures.json',
                            '-f')
    assert ret.success
