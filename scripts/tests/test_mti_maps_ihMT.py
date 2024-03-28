#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['ihMT.zip'])
tmp_dir = tempfile.TemporaryDirectory()

# Preparing once the filenames.

in_mask = os.path.join(SCILPY_HOME, 'ihMT', 'mask_resample.nii.gz')

in_mtoff_pd_json = os.path.join(SCILPY_HOME,
                                'ihMT', 'echo-1_acq-mtoff_ihmt.json')
in_mtoff_t1_json = os.path.join(SCILPY_HOME,
                                'ihMT', 'echo-1_acq-T1w_ihmt.json')

in_e1_altnp = os.path.join(SCILPY_HOME,
                           'ihMT', 'echo-1_acq-altnp_ihmt.nii.gz')
in_e2_altnp = os.path.join(SCILPY_HOME,
                           'ihMT', 'echo-2_acq-altnp_ihmt.nii.gz')
in_e3_altnp = os.path.join(SCILPY_HOME,
                           'ihMT', 'echo-3_acq-altnp_ihmt.nii.gz')

in_e1_altpn = os.path.join(SCILPY_HOME,
                           'ihMT', 'echo-1_acq-altpn_ihmt.nii.gz')
in_e2_altpn = os.path.join(SCILPY_HOME,
                           'ihMT', 'echo-2_acq-altpn_ihmt.nii.gz')
in_e3_altpn = os.path.join(SCILPY_HOME,
                           'ihMT', 'echo-3_acq-altpn_ihmt.nii.gz')

in_e1_mtoff_pd = os.path.join(SCILPY_HOME,
                              'ihMT', 'echo-1_acq-mtoff_ihmt.nii.gz')
in_e2_mtoff_pd = os.path.join(SCILPY_HOME,
                              'ihMT', 'echo-2_acq-mtoff_ihmt.nii.gz')
in_e3_mtoff_pd = os.path.join(SCILPY_HOME,
                              'ihMT', 'echo-3_acq-mtoff_ihmt.nii.gz')

in_e1_neg = os.path.join(SCILPY_HOME,
                         'ihMT', 'echo-1_acq-neg_ihmt.nii.gz')
in_e2_neg = os.path.join(SCILPY_HOME,
                         'ihMT', 'echo-2_acq-neg_ihmt.nii.gz')
in_e3_neg = os.path.join(SCILPY_HOME,
                         'ihMT', 'echo-3_acq-neg_ihmt.nii.gz')

in_e1_pos = os.path.join(SCILPY_HOME,
                         'ihMT', 'echo-1_acq-pos_ihmt.nii.gz')
in_e2_pos = os.path.join(SCILPY_HOME,
                         'ihMT', 'echo-2_acq-pos_ihmt.nii.gz')
in_e3_pos = os.path.join(SCILPY_HOME,
                         'ihMT', 'echo-3_acq-pos_ihmt.nii.gz')

in_e1_mtoff_t1 = os.path.join(SCILPY_HOME,
                              'ihMT', 'echo-1_acq-T1w_ihmt.nii.gz')
in_e2_mtoff_t1 = os.path.join(SCILPY_HOME,
                              'ihMT', 'echo-2_acq-T1w_ihmt.nii.gz')
in_e3_mtoff_t1 = os.path.join(SCILPY_HOME,
                              'ihMT', 'echo-3_acq-T1w_ihmt.nii.gz')

in_b1_map = os.path.join(SCILPY_HOME, 'ihMT', 'B1map.nii.gz')
in_b1_json = os.path.join(SCILPY_HOME, 'MT', 'sub-001_run-01_B1map.json')


def test_help_option(script_runner):
    ret = script_runner.run('scil_mti_maps_ihMT.py', '--help')
    assert ret.success


def test_execution_ihMT_no_option(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # no option
    ret = script_runner.run('scil_mti_maps_ihMT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_altnp', in_e1_altnp, in_e2_altnp,
                            in_e3_altnp,
                            '--in_altpn', in_e1_altpn, in_e2_altpn,
                            in_e3_altpn,
                            '--in_mtoff_pd', in_e1_mtoff_pd, in_e2_mtoff_pd,
                            in_e3_mtoff_pd,
                            '--in_negative', in_e1_neg, in_e2_neg, in_e3_neg,
                            '--in_positive', in_e1_pos, in_e2_pos, in_e3_pos,
                            '--in_mtoff_t1', in_e1_mtoff_t1, in_e2_mtoff_t1,
                            in_e3_mtoff_t1,
                            '--in_jsons', in_mtoff_pd_json,
                            in_mtoff_t1_json,
                            '-f')
    assert ret.success


def test_execution_ihMT_prefix(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # --out_prefix
    ret = script_runner.run('scil_mti_maps_ihMT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_altnp', in_e1_altnp, in_e2_altnp,
                            in_e3_altnp,
                            '--in_altpn', in_e1_altpn, in_e2_altpn,
                            in_e3_altpn,
                            '--in_mtoff_pd', in_e1_mtoff_pd, in_e2_mtoff_pd,
                            in_e3_mtoff_pd,
                            '--in_negative', in_e1_neg, in_e2_neg,
                            in_e3_neg,
                            '--in_positive', in_e1_pos, in_e2_pos,
                            in_e3_pos,
                            '--in_mtoff_t1', in_e1_mtoff_t1, in_e2_mtoff_t1,
                            in_e3_mtoff_t1,
                            '--in_jsons', in_mtoff_pd_json,
                            in_mtoff_t1_json,
                            '--out_prefix', 'sub_01',
                            '-f')
    assert ret.success


def test_execution_ihMT_extended(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # --extended
    ret = script_runner.run('scil_mti_maps_ihMT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_altnp', in_e1_altnp, in_e2_altnp,
                            in_e3_altnp,
                            '--in_altpn', in_e1_altpn, in_e2_altpn,
                            in_e3_altpn,
                            '--in_mtoff_pd', in_e1_mtoff_pd, in_e2_mtoff_pd,
                            in_e3_mtoff_pd,
                            '--in_negative', in_e1_neg, in_e2_neg,
                            in_e3_neg,
                            '--in_positive', in_e1_pos, in_e2_pos,
                            in_e3_pos,
                            '--in_mtoff_t1', in_e1_mtoff_t1, in_e2_mtoff_t1,
                            in_e3_mtoff_t1,
                            '--in_jsons', in_mtoff_pd_json,
                            in_mtoff_t1_json,
                            '--extended',
                            '-f')
    assert ret.success


def test_execution_ihMT_filtering(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # --filtering
    ret = script_runner.run('scil_mti_maps_ihMT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_altnp', in_e1_altnp, in_e2_altnp,
                            in_e3_altnp,
                            '--in_altpn', in_e1_altpn, in_e2_altpn,
                            in_e3_altpn,
                            '--in_mtoff_pd', in_e1_mtoff_pd, in_e2_mtoff_pd,
                            in_e3_mtoff_pd,
                            '--in_negative', in_e1_neg, in_e2_neg, in_e3_neg,
                            '--in_positive', in_e1_pos, in_e2_pos, in_e3_pos,
                            '--in_mtoff_t1', in_e1_mtoff_t1, in_e2_mtoff_t1,
                            in_e3_mtoff_t1,
                            '--in_jsons', in_mtoff_pd_json,
                            in_mtoff_t1_json,
                            '--out_prefix', 'sub-01',
                            '--filtering',
                            '-f')
    assert ret.success


def test_execution_ihMT_B1_map(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    out_b1_map = tmp_dir.name + '/B1map.nii.gz'

    # Temporary trick to have the B1 map with proper header.
    ret = script_runner.run('scil_mti_adjust_B1_header.py', in_b1_map,
                            out_b1_map, in_b1_json, '-f')

    ret = script_runner.run('scil_mti_maps_ihMT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_altnp', in_e1_altnp, in_e2_altnp,
                            in_e3_altnp,
                            '--in_altpn', in_e1_altpn, in_e2_altpn,
                            in_e3_altpn,
                            '--in_mtoff_pd', in_e1_mtoff_pd, in_e2_mtoff_pd,
                            in_e3_mtoff_pd,
                            '--in_negative', in_e1_neg, in_e2_neg, in_e3_neg,
                            '--in_positive', in_e1_pos, in_e2_pos, in_e3_pos,
                            '--in_mtoff_t1', in_e1_mtoff_t1, in_e2_mtoff_t1,
                            in_e3_mtoff_t1,
                            '--out_prefix', 'sub-01',
                            '--in_B1_map', out_b1_map,
                            '--B1_correction_method', 'empiric',
                            '--in_jsons', in_mtoff_pd_json,
                            in_mtoff_t1_json,
                            '-f')
    assert ret.success


def test_execution_ihMT_B1_no_T1(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    out_b1_map = tmp_dir.name + '/B1map.nii.gz'

    # Temporary trick to have the B1 map with proper header.
    ret = script_runner.run('scil_mti_adjust_B1_header.py', in_b1_map,
                            out_b1_map, in_b1_json, '-f')

    ret = script_runner.run('scil_mti_maps_ihMT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_altnp', in_e1_altnp, in_e2_altnp,
                            in_e3_altnp,
                            '--in_altpn', in_e1_altpn, in_e2_altpn,
                            in_e3_altpn,
                            '--in_mtoff_pd', in_e1_mtoff_pd, in_e2_mtoff_pd,
                            in_e3_mtoff_pd,
                            '--in_negative', in_e1_neg, in_e2_neg, in_e3_neg,
                            '--in_positive', in_e1_pos, in_e2_pos, in_e3_pos,
                            '--in_B1_map', out_b1_map,
                            '--B1_correction_method', 'empiric',
                            '--in_jsons', in_mtoff_pd_json,
                            in_mtoff_t1_json,
                            '-f')
    assert ret.success


def test_execution_ihMT_wrong_echoes(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_mti_maps_ihMT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_altnp', in_e1_altnp, in_e2_altnp,
                            in_e3_altnp,
                            '--in_altpn', in_e1_altpn, in_e2_altpn,
                            in_e3_altpn,
                            '--in_mtoff_pd', in_e1_mtoff_pd, in_e2_mtoff_pd,
                            in_e3_mtoff_pd,
                            '--in_negative', in_e1_neg, in_e2_neg, in_e3_neg,
                            '--in_positive', in_e1_pos, in_e2_pos, in_e3_pos,
                            '--in_mtoff_t1', in_e1_mtoff_t1, in_e2_mtoff_t1,
                            '--in_jsons', in_mtoff_pd_json,
                            in_mtoff_t1_json,
                            '-f')
    assert (not ret.success)


def test_execution_ihMT_B1_no_fit(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    out_b1_map = tmp_dir.name + '/B1map.nii.gz'

    # Temporary trick to have the B1 map with proper header.
    ret = script_runner.run('scil_mti_adjust_B1_header.py', in_b1_map,
                            out_b1_map, in_b1_json, '-f')

    ret = script_runner.run('scil_mti_maps_ihMT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_altnp', in_e1_altnp, in_e2_altnp,
                            in_e3_altnp,
                            '--in_altpn', in_e1_altpn, in_e2_altpn,
                            in_e3_altpn,
                            '--in_mtoff_pd', in_e1_mtoff_pd, in_e2_mtoff_pd,
                            in_e3_mtoff_pd,
                            '--in_negative', in_e1_neg, in_e2_neg, in_e3_neg,
                            '--in_positive', in_e1_pos, in_e2_pos, in_e3_pos,
                            '--in_mtoff_t1', in_e1_mtoff_t1, in_e2_mtoff_t1,
                            in_e3_mtoff_t1,
                            '--out_prefix', 'sub-01',
                            '--in_B1_map', out_b1_map,
                            '--B1_correction_method', 'model_based',
                            '--in_jsons', in_mtoff_pd_json,
                            in_mtoff_t1_json,
                            '-f')
    assert (not ret.success)


def test_execution_ihMT_single_echo(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_mti_maps_ihMT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_altnp', in_e1_altnp,
                            '--in_altpn', in_e1_altpn,
                            '--in_mtoff_pd', in_e1_mtoff_pd,
                            '--in_negative', in_e1_neg,
                            '--in_positive', in_e1_pos,
                            '--in_mtoff_t1', in_e1_mtoff_t1,
                            '--out_prefix', 'sub_01',
                            '--in_jsons', in_mtoff_pd_json,
                            in_mtoff_t1_json, '-f')
    assert ret.success


def test_execution_ihMT_acq_params(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_mti_maps_ihMT.py', tmp_dir.name,
                            '--mask', in_mask,
                            '--in_altnp', in_e1_altnp,
                            '--in_altpn', in_e1_altpn,
                            '--in_mtoff_pd', in_e1_mtoff_pd,
                            '--in_negative', in_e1_neg,
                            '--in_positive', in_e1_pos,
                            '--in_mtoff_t1', in_e1_mtoff_t1,
                            '--out_prefix', 'sub_01',
                            '--in_acq_parameters', '15', '15', '0.1', '0.1',
                            '-f')
    assert ret.success
