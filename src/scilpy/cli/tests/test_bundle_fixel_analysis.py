#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.io.streamlines import save_tractogram

fetch_data(get_testing_files_dict(), keys=['commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_bundle_fixel_analysis', '--help'])
    assert ret.success


def test_default_parameters(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_peaks = os.path.join(SCILPY_HOME, 'commit_amico', 'peaks.nii.gz')
    in_bundle = os.path.join(SCILPY_HOME, 'commit_amico', 'tracking.trk')

    # Using multiprocessing in this test, single in following tests.
    ret = script_runner.run(['scil_bundle_fixel_analysis', in_peaks,
                             '--in_bundles', in_bundle,
                             '--processes', '4', '-f'])
    assert ret.success
    fd_name = os.path.join('fixel_analysis',
                           'fixel_density_maps_voxel-norm.nii.gz')
    v_name = os.path.join('fixel_analysis',
                          'fixel_density_maps_v-norm.nii.gz')
    assert os.path.isfile(fd_name)
    assert not os.path.isfile(v_name)


def test_all_parameters(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_peaks = os.path.join(SCILPY_HOME, 'commit_amico', 'peaks.nii.gz')
    in_bundle = os.path.join(SCILPY_HOME, 'commit_amico', 'tracking.trk')

    ret = script_runner.run(['scil_bundle_fixel_analysis', in_peaks,
                             '--in_bundles', in_bundle,
                             '--in_bundles_names', 'test',
                             '--abs_thr', '5',
                             '--rel_thr', '0.05',
                             '--norm', 'fixel',
                             '--split_bundles', '--split_fixels',
                             '--single_bundle',
                             '--processes', '1', '-f'])
    assert ret.success


def test_multiple_norm(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_peaks = os.path.join(SCILPY_HOME, 'commit_amico', 'peaks.nii.gz')
    in_bundle = os.path.join(SCILPY_HOME, 'commit_amico', 'tracking.trk')

    ret = script_runner.run(['scil_bundle_fixel_analysis', in_peaks,
                             '--in_bundles', in_bundle,
                             '--in_bundles_names', 'test',
                             '--abs_thr', '5',
                             '--rel_thr', '0.05',
                             '--norm', 'fixel', 'none', 'voxel',
                             '--split_bundles', '--split_fixels',
                             '--single_bundle',
                             '--out_dir', '.',
                             '--processes', '1', '-f'])
    assert ret.success
    assert os.path.isfile('bundles_LUT.txt')
    for n in ['voxel', 'fixel', 'none']:
        assert os.path.isfile('fixel_density_maps_{}-norm.nii.gz'.format(n))
        assert os.path.isfile('fixel_density_map_{}-norm_f1.nii.gz'.format(n))
        assert os.path.isfile('fixel_density_map_{}-norm_f2.nii.gz'.format(n))
        assert os.path.isfile('fixel_density_map_{}-norm_f3.nii.gz'.format(n))
        assert os.path.isfile('fixel_density_map_{}-norm_f4.nii.gz'.format(n))
        assert os.path.isfile('fixel_density_map_{}-norm_f5.nii.gz'.format(n))
        assert os.path.isfile('fixel_density_map_{}-'
                              'norm_test.nii.gz'.format(n))
        assert os.path.isfile('nb_bundles_per_fixel_{}-norm.nii.gz'.format(n))
        assert os.path.isfile('nb_bundles_per_voxel_{}-norm.nii.gz'.format(n))
        assert os.path.isfile('single_bundle_mask_{}-'
                              'norm_WM.nii.gz'.format(n))
        assert os.path.isfile('single_bundle_mask_{}-'
                              'norm_test.nii.gz'.format(n))

    assert os.path.isfile('voxel_density_maps_voxel-norm.nii.gz')
    assert not os.path.isfile('voxel_density_maps_fixel-norm.nii.gz')
    assert os.path.isfile('voxel_density_maps_none-norm.nii.gz')


def test_non_ras_bundle_fixel_analysis(script_runner, tmp_path):
    # Non-RAS affine (LAS orientation with offset)
    affine = np.array([
        [-2.0, 0.0, 0.0, 30.0],
        [0.0, 2.0, 0.0, 10.0],
        [0.0, 0.0, 2.0, 5.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # Peak at voxel (2, 3, 4) pointing in world X
    peaks_data = np.zeros((8, 8, 8, 15), dtype=np.float32)
    peaks_data[2, 3, 4, 0:3] = [1.0, 0.0, 0.0]

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

    out_dir = str(tmp_path / "out")
    ret = script_runner.run(['scil_bundle_fixel_analysis', peaks_path,
                             '--in_bundles', bundle_path,
                             '--out_dir', out_dir,
                             '--processes', '1', '-f'])
    assert ret.success
    out_map_path = os.path.join(out_dir,
                                "fixel_density_maps_voxel-norm.nii.gz")
    assert os.path.isfile(out_map_path)
    res_img = nib.load(out_map_path)
    res_data = res_img.get_fdata()
    # Fixel 0 at voxel (2, 3, 4) must have density > 0
    assert res_data[2, 3, 4, 0, 0] > 0.0


def test_empty_bundle_fixel_analysis(script_runner, tmp_path):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    peaks_data = np.zeros((8, 8, 8, 15), dtype=np.float32)
    peaks_path = str(tmp_path / "peaks.nii.gz")
    peaks_img = nib.Nifti1Image(peaks_data, affine)
    nib.save(peaks_img, peaks_path)

    sft = StatefulTractogram([], peaks_img, Space.VOX)
    bundle_path = str(tmp_path / "empty_bundle.trk")
    save_tractogram(sft, bundle_path, False)

    out_dir = str(tmp_path / "out_empty")
    ret = script_runner.run(['scil_bundle_fixel_analysis', peaks_path,
                             '--in_bundles', bundle_path,
                             '--out_dir', out_dir,
                             '--processes', '1', '-f'])
    assert ret.success
    out_map_path = os.path.join(out_dir,
                                "fixel_density_maps_voxel-norm.nii.gz")
    assert os.path.isfile(out_map_path)
    res_data = nib.load(out_map_path).get_fdata()
    assert np.all(res_data == 0.0)


def test_out_of_bounds_bundle_fixel_analysis(script_runner, tmp_path):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    peaks_data = np.zeros((8, 8, 8, 15), dtype=np.float32)
    peaks_path = str(tmp_path / "peaks_oob.nii.gz")
    peaks_img = nib.Nifti1Image(peaks_data, affine)
    nib.save(peaks_img, peaks_path)

    # Streamline completely outside bounding box
    streamline = np.array([
        [100.0, 100.0, 100.0],
        [105.0, 105.0, 105.0]
    ])
    sft = StatefulTractogram([streamline], peaks_img, Space.VOX)
    bundle_path = str(tmp_path / "oob_bundle.trk")
    save_tractogram(sft, bundle_path, False, bbox_valid_check=False)

    out_dir = str(tmp_path / "out_oob")
    ret = script_runner.run(['scil_bundle_fixel_analysis', peaks_path,
                             '--in_bundles', bundle_path,
                             '--out_dir', out_dir,
                             '--processes', '1', '-f'])
    assert ret.success
    out_map_path = os.path.join(out_dir,
                                "fixel_density_maps_voxel-norm.nii.gz")
    assert os.path.isfile(out_map_path)
    res_data = nib.load(out_map_path).get_fdata()
    assert np.all(res_data == 0.0)
