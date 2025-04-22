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
    header = nib.nifti2.Nifti2Header()
    extra = {
        'affine': affine,
        'dimensions': (15, 15, 15),
        'voxel_size': 1.,
        'voxel_order': "RAS"
    }
    mask_img = nib.nifti2.Nifti2Image(mask, affine, header, extra)

    config = {
        'step_size': 0.001,
        'blur_radius': 0.001,
        'nb_fibertubes': 2,
        'nb_seeds_per_fibertube': 1,
    }

    sft_fibertubes = StatefulTractogram(streamlines, mask_img,
                                        space=Space.VOX,
                                        origin=Origin.NIFTI)
    sft_fibertubes.data_per_streamline = {
        "diameters": [0.002, 0.001]
    }
    sft_tracking = StatefulTractogram(streamlines, mask_img,
                                      space=Space.VOX,
                                      origin=Origin.NIFTI)
    sft_tracking.data_per_streamline = {
        "seeds": [streamlines[0][0], streamlines[1][-1]],
        "seed_ids": [0., 1.]
    }

    save_tractogram(sft_fibertubes, 'fibertubes.trk', True)
    save_tractogram(sft_tracking, 'tracking.trk', True)

    with open('config.json', 'w') as file:
        json.dump(config, file, indent=True)


def test_help_option(script_runner):
    ret = script_runner.run('scil_fibertube_score_tractogram.py', '--help')
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_score_tractogram.py',
                            'fibertubes.trk', 'tracking.trk', 'config.json',
                            'metrics.json', '--save_error_tractogram', '-f')
    assert ret.success
