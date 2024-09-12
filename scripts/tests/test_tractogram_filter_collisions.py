#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import numpy as np
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

    sft = StatefulTractogram(streamlines, mask_img, Space.VOX, Origin.NIFTI)
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


def test_execution_filtering_save_colliding(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    init_data()

    diameters = [5, 1]
    np.savetxt('diameters.txt', diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            'tractogram.trk', 'diameters.txt', 'clean.trk',
                            '--save_colliding', '-f')

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
    diameters = [1, 1]
    np.savetxt('diameters.txt', diameters)

    ret = script_runner.run('scil_tractogram_filter_collisions.py',
                            'tractogram.trk', 'diameters.txt', 'clean.trk',
                            '--out_metrics', 'metrics.txt', '-f')

    assert ret.success
