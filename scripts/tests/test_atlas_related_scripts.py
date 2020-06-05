#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict())
tmp_dir = tempfile.TemporaryDirectory()


def test_remove_labels(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_atlas = os.path.join(get_home(), 'atlas',
                               'atlas_freesurfer_v2.nii.gz')
    ret = script_runner.run('scil_remove_labels.py', input_atlas,
                            'atlas_freesurfer_v2_no_brainstem.nii.gz',
                            '-i', '173', '174', '175')
    return ret.success


def test_split_label(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_atlas = os.path.join(get_home(), 'atlas',
                               'atlas_freesurfer_v2.nii.gz')
    ret = script_runner.run('scil_split_volume_by_ids.py', input_atlas,
                            '--out_prefix', 'brainstem', '-r', '173-175')
    return ret.success


def test_math_add(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_image_math.py', 'addition',
                            'brainstem_17*', 'brainstem.nii.gz')
    return ret.success


def test_math_low_thresh(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_image_math.py', 'lower_threshold',
                            'brainstem.nii.gz', '1',
                            'brainstem_bin.nii.gz')
    return ret.success


def test_math_low_mult(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_image_math.py', 'multiplication',
                            'brainstem.nii.gz', '16',
                            'brainstem_unified.nii.gz')
    return ret.success


def test_math_combine(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_atlas = os.path.join(get_home(), 'atlas',
                               'atlas_freesurfer_v2.nii.gz')
    ret = script_runner.run('scil_combine_labels.py',
                            'atlas_freesurfer_v2_single_brainstem.nii.gz',
                            '-v', input_atlas, '{1..2035}',
                            '-v', 'brainstem.nii.gz', '16')
    return ret.success


def test_math_dilate(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_dilate_labels.py',
                            'atlas_freesurfer_v2_single_brainstem.nii.gz',
                            'atlas_freesurfer_v2_single_brainstem_dil.nii.gz',
                            '--processes', '1', '--distance', '2')
    return ret.success
