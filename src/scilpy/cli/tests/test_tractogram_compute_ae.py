#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.io.streamlines import save_tractogram

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_tractogram_compute_ae',
                            '--help'])
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'commit_amico', 'tracking.trk')
    in_peaks = os.path.join(SCILPY_HOME, 'commit_amico', 'peaks.nii.gz')

    ret = script_runner.run(['scil_tractogram_compute_ae', in_bundle, in_peaks,
                             'out_bundle.trk', '--dpp_key', 'AE',
                             '--save_mean_map', 'out_map.nii.gz',
                             '--save_as_color', '--processes', '4',
                             '--cmap_max', '70'])
    assert ret.success


def test_non_ras_tractogram_compute_ae(script_runner, tmp_path):
    # Non-RAS affine (LAS orientation with offset)
    affine = np.array([
        [-2.0, 0.0, 0.0, 30.0],
        [0.0, 2.0, 0.0, 10.0],
        [0.0, 0.0, 2.0, 5.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # Peak at voxel (2, 3, 4) pointing in world X
    peaks_data = np.zeros((8, 8, 8, 3), dtype=np.float32)
    peaks_data[2, 3, 4] = [1.0, 0.0, 0.0]

    peaks_path = str(tmp_path / "peaks_las.nii.gz")
    peaks_img = nib.Nifti1Image(peaks_data, affine)
    nib.save(peaks_img, peaks_path)

    # Streamline through voxel (2, 3, 4) in world space
    vox_center = np.array([2.0, 3.0, 4.0, 1.0])
    world_center = affine @ vox_center
    streamline = np.array([
        world_center[:3] - np.array([0.4, 0.0, 0.0]),
        world_center[:3] + np.array([0.4, 0.0, 0.0])
    ])
    sft = StatefulTractogram([streamline], peaks_img, Space.RASMM)
    bundle_path = str(tmp_path / "bundle.trk")
    save_tractogram(sft, bundle_path, False)

    out_bundle = str(tmp_path / "out_bundle.trk")
    out_map = str(tmp_path / "out_map.nii.gz")
    ret = script_runner.run(['scil_tractogram_compute_ae', bundle_path,
                             peaks_path, out_bundle, '--dpp_key', 'AE',
                             '--save_mean_map', out_map, '--processes', '1'])
    assert ret.success

    # Streamline segment is aligned with peak, so AE must be near 0
    sft_out = load_tractogram(out_bundle, 'same')
    ae_values = sft_out.data_per_point['AE'][0]
    assert np.allclose(ae_values[0], 0.0, atol=1e-3)

    # Mean map at voxel (2, 3, 4) should have AE near 0
    map_img = nib.load(out_map)
    map_data = map_img.get_fdata()
    assert np.isclose(map_data[2, 3, 4], 0.0, atol=1e-3)


def test_empty_tractogram_compute_ae(script_runner, tmp_path):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    peaks_data = np.zeros((8, 8, 8, 15), dtype=np.float32)
    peaks_path = str(tmp_path / "peaks.nii.gz")
    peaks_img = nib.Nifti1Image(peaks_data, affine)
    nib.save(peaks_img, peaks_path)

    sft = StatefulTractogram([], peaks_img, Space.VOX)
    bundle_path = str(tmp_path / "empty_bundle.trk")
    save_tractogram(sft, bundle_path, False)

    out_bundle = str(tmp_path / "out_bundle.trk")
    out_map = str(tmp_path / "out_map.nii.gz")
    ret = script_runner.run(['scil_tractogram_compute_ae', bundle_path,
                             peaks_path, out_bundle, '--dpp_key', 'AE',
                             '--save_mean_map', out_map, '--processes', '1'])
    assert ret.success
    assert os.path.isfile(out_bundle)
    assert os.path.isfile(out_map)


def test_empty_tractogram_compute_ae_save_as_color(script_runner, tmp_path):
    """
    --save_as_color on an empty tractogram used to crash with an
    IndexError (add_data_as_color_dpp indexing into an empty array).
    """
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    peaks_data = np.zeros((8, 8, 8, 15), dtype=np.float32)
    peaks_path = str(tmp_path / "peaks.nii.gz")
    peaks_img = nib.Nifti1Image(peaks_data, affine)
    nib.save(peaks_img, peaks_path)

    sft = StatefulTractogram([], peaks_img, Space.VOX)
    bundle_path = str(tmp_path / "empty_bundle_color.trk")
    save_tractogram(sft, bundle_path, False)

    out_bundle = str(tmp_path / "out_bundle_color.trk")
    ret = script_runner.run(['scil_tractogram_compute_ae', bundle_path,
                             peaks_path, out_bundle, '--dpp_key', 'AE',
                             '--save_as_color', '--processes', '1'])
    assert ret.success
    assert os.path.isfile(out_bundle)
