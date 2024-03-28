#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['MT.zip'])
tmp_dir = tempfile.TemporaryDirectory()


# Preparing once the filenames.
in_mask = os.path.join(SCILPY_HOME, 'MT', 'mask.nii.gz')

in_mtoff_json = os.path.join(SCILPY_HOME,
                             'MT', 'sub-001_echo-1_acq-mtoff_mtsat.json')
in_t1w_json = os.path.join(SCILPY_HOME,
                           'MT', 'sub-001_echo-1_acq-t1w_mtsat.json')

in_e1_mtoff = os.path.join(SCILPY_HOME,
                           'MT', 'sub-001_echo-1_acq-mtoff_mtsat.nii.gz')
in_e2_mtoff = os.path.join(SCILPY_HOME,
                           'MT', 'sub-001_echo-2_acq-mtoff_mtsat.nii.gz')
in_e3_mtoff = os.path.join(SCILPY_HOME,
                           'MT', 'sub-001_echo-3_acq-mtoff_mtsat.nii.gz')
in_e4_mtoff = os.path.join(SCILPY_HOME,
                           'MT', 'sub-001_echo-4_acq-mtoff_mtsat.nii.gz')
in_e5_mtoff = os.path.join(SCILPY_HOME,
                           'MT', 'sub-001_echo-5_acq-mtoff_mtsat.nii.gz')

in_e1_mton = os.path.join(SCILPY_HOME,
                          'MT', 'sub-001_echo-1_acq-mton_mtsat.nii.gz')
in_e2_mton = os.path.join(SCILPY_HOME,
                          'MT', 'sub-001_echo-2_acq-mton_mtsat.nii.gz')
in_e3_mton = os.path.join(SCILPY_HOME,
                          'MT', 'sub-001_echo-3_acq-mton_mtsat.nii.gz')
in_e4_mton = os.path.join(SCILPY_HOME,
                          'MT', 'sub-001_echo-4_acq-mton_mtsat.nii.gz')
in_e5_mton = os.path.join(SCILPY_HOME,
                          'MT', 'sub-001_echo-5_acq-mton_mtsat.nii.gz')

in_e1_t1w = os.path.join(SCILPY_HOME,
                         'MT', 'sub-001_echo-1_acq-t1w_mtsat.nii.gz')
in_e2_t1w = os.path.join(SCILPY_HOME,
                         'MT', 'sub-001_echo-2_acq-t1w_mtsat.nii.gz')
in_e3_t1w = os.path.join(SCILPY_HOME,
                         'MT', 'sub-001_echo-3_acq-t1w_mtsat.nii.gz')
in_e4_t1w = os.path.join(SCILPY_HOME,
                         'MT', 'sub-001_echo-4_acq-t1w_mtsat.nii.gz')
in_e5_t1w = os.path.join(SCILPY_HOME,
                         'MT', 'sub-001_echo-5_acq-t1w_mtsat.nii.gz')

in_b1_map = os.path.join(SCILPY_HOME, 'MT', 'sub-001_run-01_B1map.nii.gz')
in_b1_json = os.path.join(SCILPY_HOME, 'MT', 'sub-001_run-01_B1map.json')


def test_help_option(script_runner):
    ret = script_runner.run('scil_mti_maps_MT.py', '--help')
    assert ret.success


def test_execution_MT_no_option(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # no option
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_mtoff_pd', in_e1_mtoff, in_e2_mtoff,
                            in_e3_mtoff, in_e4_mtoff, in_e5_mtoff,
                            '--in_positive', in_e1_mton, in_e2_mton,
                            in_e3_mton, in_e4_mton, in_e5_mton,
                            '--in_mtoff_t1', in_e1_t1w, in_e2_t1w, in_e3_t1w,
                            in_e4_t1w, in_e5_t1w,
                            '--in_jsons', in_mtoff_json, in_t1w_json,
                            '-f')
    assert ret.success


def test_execution_MT_prefix(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # --out_prefix
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_mtoff_pd', in_e1_mtoff, in_e2_mtoff,
                            in_e3_mtoff, in_e4_mtoff, in_e5_mtoff,
                            '--in_positive', in_e1_mton, in_e2_mton,
                            in_e3_mton, in_e4_mton, in_e5_mton,
                            '--in_mtoff_t1', in_e1_t1w, in_e2_t1w, in_e3_t1w,
                            in_e4_t1w, in_e5_t1w,
                            '--in_jsons', in_mtoff_json, in_t1w_json,
                            '--out_prefix', 'sub_01',
                            '-f')
    assert ret.success


def test_execution_MT_extended(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # --extended
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_mtoff_pd', in_e1_mtoff, in_e2_mtoff,
                            in_e3_mtoff, in_e4_mtoff, in_e5_mtoff,
                            '--in_positive', in_e1_mton, in_e2_mton,
                            in_e3_mton, in_e4_mton, in_e5_mton,
                            '--in_mtoff_t1', in_e1_t1w, in_e2_t1w, in_e3_t1w,
                            in_e4_t1w, in_e5_t1w,
                            '--in_jsons', in_mtoff_json, in_t1w_json,
                            '--extended',
                            '-f')
    assert ret.success


def test_execution_MT_filtering(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # --filtering
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_mtoff_pd', in_e1_mtoff, in_e2_mtoff,
                            in_e3_mtoff, in_e4_mtoff, in_e5_mtoff,
                            '--in_positive', in_e1_mton, in_e2_mton,
                            in_e3_mton, in_e4_mton, in_e5_mton,
                            '--in_mtoff_t1', in_e1_t1w, in_e2_t1w, in_e3_t1w,
                            in_e4_t1w, in_e5_t1w,
                            '--in_jsons', in_mtoff_json, in_t1w_json,
                            '--filtering',
                            '-f')
    assert ret.success


def test_execution_MT_B1_map(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    out_b1_map = tmp_dir.name + '/B1map.nii.gz'

    # Temporary trick to have the B1 map with proper header.
    ret = script_runner.run('scil_mti_adjust_B1_header.py', in_b1_map,
                            out_b1_map, in_b1_json, '-f')

    # --in_B1_map
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_mtoff_pd', in_e1_mtoff, in_e2_mtoff,
                            in_e3_mtoff, in_e4_mtoff, in_e5_mtoff,
                            '--in_positive', in_e1_mton, in_e2_mton,
                            in_e3_mton, in_e4_mton, in_e5_mton,
                            '--in_negative', in_e1_mton, in_e2_mton,
                            in_e3_mton, in_e4_mton, in_e5_mton,
                            '--in_mtoff_t1', in_e1_t1w, in_e2_t1w, in_e3_t1w,
                            in_e4_t1w, in_e5_t1w,
                            '--in_jsons', in_mtoff_json, in_t1w_json,
                            '--in_B1_map', out_b1_map,
                            '--B1_correction_method', 'empiric',
                            '--out_prefix', 'sub-01',
                            '-f')
    assert ret.success


def test_execution_MT_wrong_echoes(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # Wrong number of echoes for negative
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_mtoff_pd', in_e1_mtoff, in_e2_mtoff,
                            in_e3_mtoff, in_e4_mtoff, in_e5_mtoff,
                            '--in_positive', in_e1_mton, in_e2_mton,
                            in_e3_mton, in_e4_mton, in_e5_mton,
                            '--in_negative', in_e1_mton, in_e2_mton,
                            in_e3_mton, in_e4_mton,
                            '--in_mtoff_t1', in_e1_t1w, in_e2_t1w, in_e3_t1w,
                            in_e4_t1w, in_e5_t1w,
                            '--in_jsons', in_mtoff_json, in_t1w_json,
                            '-f')
    assert (not ret.success)


def test_execution_MT_single_echoe(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # Single echoe
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_mtoff_pd', in_e1_mtoff,
                            '--in_positive', in_e1_mton,
                            '--in_negative', in_e1_mton,
                            '--in_mtoff_t1', in_e1_t1w,
                            '--in_jsons', in_mtoff_json, in_t1w_json,
                            '-f')
    assert ret.success


def test_execution_MT_B1_not_T1(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    out_b1_map = tmp_dir.name + '/B1map.nii.gz'

    # Temporary trick to have the B1 map with proper header.
    ret = script_runner.run('scil_mti_adjust_B1_header.py', in_b1_map,
                            out_b1_map, in_b1_json, '-f')

    # B1 no T1 should raise warning.
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_mtoff_pd', in_e1_mtoff,
                            '--in_positive', in_e1_mton,
                            '--in_negative', in_e1_mton,
                            '--in_jsons', in_mtoff_json, in_t1w_json,
                            '--in_B1_map', out_b1_map,
                            '--B1_correction_method', 'empiric',
                            '-f')
    assert ret.success


def test_execution_MT_B1_no_fit(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    out_b1_map = tmp_dir.name + '/B1map.nii.gz'

    # Temporary trick to have the B1 map with proper header.
    ret = script_runner.run('scil_mti_adjust_B1_header.py', in_b1_map,
                            out_b1_map, in_b1_json, '-f')

    # B1 model_based but no fit values
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_mtoff_pd', in_e1_mtoff,
                            '--in_positive', in_e1_mton,
                            '--in_negative', in_e1_mton,
                            '--in_mtoff_t1', in_e1_t1w,
                            '--in_jsons', in_mtoff_json, in_t1w_json,
                            '--in_B1_map', out_b1_map,
                            '--B1_correction_method', 'model_based',
                            '-f')
    assert (not ret.success)


def test_execution_MT_acq_params(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # Acquisition parameters
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_mtoff_pd', in_e1_mtoff,
                            '--in_positive', in_e1_mton,
                            '--in_negative', in_e1_mton,
                            '--in_mtoff_t1', in_e1_t1w,
                            '--in_acq_parameters', "15", "15", "0.1", "0.1",
                            '-f')
    assert ret.success
