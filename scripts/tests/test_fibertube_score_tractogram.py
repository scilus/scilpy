#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import tempfile
import nibabel as nib

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.io.streamlines import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractograms.zip'])
tmp_dir = tempfile.TemporaryDirectory()

def init_data():
    streamlines = [[[4., 0., 4.], [4., 4., 8.], [6., 8., 8.], [12., 10., 8.], [4.,6., 6.]],
                [[6., 6., 6.], [8., 8., 8.]]]

    in_mask = os.path.join(SCILPY_HOME, 'tracking',
                            'seeding_mask.nii.gz')
    mask_img = nib.load(in_mask)

    config = {
        'step_size': 0,
        'blur_radius': 0,
        'nb_fibers': 2,
        'nb_seeds_per_fiber': 1,
    }

    sft = StatefulTractogram(streamlines, mask_img, Space.VOX, Origin.NIFTI)
    save_tractogram(sft, 'tracking.trk', True)
    json.dump(config, 'config.txt', indent=True)

    sft.data_per_streamline = {
        "diameters": [0.002, 0.001]
    }
    save_tractogram(sft, 'fibertubes.trk', True)


def test_help_option(script_runner):
    ret = script_runner.run('scil_fibertube_score_tractogram.py', '--help')
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()
    ret = script_runner.run('scil_fibertube_score_tractogram.py',
                            'fibertube.trk', 'tracking.trk', 'config.txt',
                            'metrix.txt', '--save_error_tractogram')

    assert ret.success


