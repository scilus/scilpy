#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['plot.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_viz_volume_scatterplot.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_x = os.path.join(SCILPY_HOME, 'plot',
                        'fa.nii.gz')
    in_y = os.path.join(SCILPY_HOME, 'plot',
                        'ad.nii.gz')
    ret = script_runner.run('scil_viz_volume_scatterplot.py', in_x, in_y,
                            'scatter_plot.png')
    assert ret.success


def test_execution_processing_bin_mask(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_x = os.path.join(SCILPY_HOME, 'plot',
                        'fa.nii.gz')
    in_y = os.path.join(SCILPY_HOME, 'plot',
                        'ad.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'plot',
                           'mask_wm.nii.gz')
    ret = script_runner.run('scil_viz_volume_scatterplot.py', in_x, in_y,
                            'scatter_plot_m.png', '--in_bin_mask', in_mask)
    assert ret.success


def test_execution_processing_prob_map(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_x = os.path.join(SCILPY_HOME, 'plot',
                        'fa.nii.gz')
    in_y = os.path.join(SCILPY_HOME, 'plot',
                        'ad.nii.gz')
    in_prob_1 = os.path.join(SCILPY_HOME, 'plot',
                             'map_wm.nii.gz')
    in_prob_2 = os.path.join(SCILPY_HOME, 'plot',
                             'map_gm.nii.gz')
    ret = script_runner.run('scil_viz_volume_scatterplot.py', in_x, in_y,
                            'scatter_plot_prob.png',
                            '--in_prob_maps', in_prob_1, in_prob_2)
    assert ret.success


def test_execution_processing_atlas(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_x = os.path.join(SCILPY_HOME, 'plot',
                        'fa.nii.gz')
    in_y = os.path.join(SCILPY_HOME, 'plot',
                        'ad.nii.gz')
    in_atlas = os.path.join(SCILPY_HOME, 'plot',
                            'atlas_brainnetome.nii.gz')
    atlas_lut = os.path.join(SCILPY_HOME, 'plot',
                             'atlas_brainnetome.json')
    ret = script_runner.run('scil_viz_volume_scatterplot.py', in_x, in_y,
                            'scatter_plot', '--in_atlas', in_atlas,
                            '--atlas_lut', atlas_lut)
    assert ret.success


def test_execution_processing_atlas_folder(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_x = os.path.join(SCILPY_HOME, 'plot',
                        'fa.nii.gz')
    in_y = os.path.join(SCILPY_HOME, 'plot',
                        'ad.nii.gz')
    in_atlas = os.path.join(SCILPY_HOME, 'plot',
                            'atlas_brainnetome.nii.gz')
    atlas_lut = os.path.join(SCILPY_HOME, 'plot',
                             'atlas_brainnetome.json')
    ret = script_runner.run('scil_viz_volume_scatterplot.py', in_x, in_y,
                            'scatter_plot', '--in_atlas', in_atlas,
                            '--atlas_lut', atlas_lut,
                            '--in_folder')
    assert ret.success


def test_execution_processing_atlas_folder_specific_label(script_runner,
                                                          monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_x = os.path.join(SCILPY_HOME, 'plot',
                        'fa.nii.gz')
    in_y = os.path.join(SCILPY_HOME, 'plot',
                        'ad.nii.gz')
    in_atlas = os.path.join(SCILPY_HOME, 'plot',
                            'atlas_brainnetome.nii.gz')
    atlas_lut = os.path.join(SCILPY_HOME, 'plot',
                             'atlas_brainnetome.json')
    ret = script_runner.run('scil_viz_volume_scatterplot.py', in_x, in_y,
                            'scatter_plot', '--in_atlas', in_atlas,
                            '--atlas_lut', atlas_lut,
                            '--specific_label', '2', '5', '7',
                            '--in_folder')
    assert ret.success
