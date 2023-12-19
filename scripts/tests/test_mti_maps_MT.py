#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['MT.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_mti_maps_MT.py', '--help')
    assert ret.success


def test_execution_MT_no_option(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    in_mask = os.path.join(get_home(), 'MT', 'mask.nii.gz')

    in_e1_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-1_acq-mtoff_mtsat.nii.gz')
    in_e2_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-2_acq-mtoff_mtsat.nii.gz')
    in_e3_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-3_acq-mtoff_mtsat.nii.gz')
    in_e4_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-4_acq-mtoff_mtsat.nii.gz')
    in_e5_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-5_acq-mtoff_mtsat.nii.gz')

    in_e1_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-1_acq-mton_mtsat.nii.gz')
    in_e2_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-2_acq-mton_mtsat.nii.gz')
    in_e3_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-3_acq-mton_mtsat.nii.gz')
    in_e4_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-4_acq-mton_mtsat.nii.gz')
    in_e5_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-5_acq-mton_mtsat.nii.gz')

    in_e1_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-1_acq-t1w_mtsat.nii.gz')
    in_e2_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-2_acq-t1w_mtsat.nii.gz')
    in_e3_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-3_acq-t1w_mtsat.nii.gz')
    in_e4_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-4_acq-t1w_mtsat.nii.gz')
    in_e5_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-5_acq-t1w_mtsat.nii.gz')

    # no option
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            in_mask,
                            '--in_mtoff', in_e1_mtoff, in_e2_mtoff,
                            in_e3_mtoff, in_e4_mtoff, in_e5_mtoff,
                            '--in_mton', in_e1_mton, in_e2_mton, in_e3_mton,
                            in_e4_mton, in_e5_mton,
                            '--in_t1w', in_e1_t1w, in_e2_t1w, in_e3_t1w,
                            in_e4_t1w, in_e5_t1w,
                            '-f')
    assert ret.success


def test_execution_MT_prefix(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    in_mask = os.path.join(get_home(), 'MT', 'mask.nii.gz')

    in_e1_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-1_acq-mtoff_mtsat.nii.gz')
    in_e2_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-2_acq-mtoff_mtsat.nii.gz')
    in_e3_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-3_acq-mtoff_mtsat.nii.gz')
    in_e4_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-4_acq-mtoff_mtsat.nii.gz')
    in_e5_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-5_acq-mtoff_mtsat.nii.gz')

    in_e1_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-1_acq-mton_mtsat.nii.gz')
    in_e2_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-2_acq-mton_mtsat.nii.gz')
    in_e3_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-3_acq-mton_mtsat.nii.gz')
    in_e4_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-4_acq-mton_mtsat.nii.gz')
    in_e5_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-5_acq-mton_mtsat.nii.gz')

    in_e1_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-1_acq-t1w_mtsat.nii.gz')
    in_e2_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-2_acq-t1w_mtsat.nii.gz')
    in_e3_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-3_acq-t1w_mtsat.nii.gz')
    in_e4_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-4_acq-t1w_mtsat.nii.gz')
    in_e5_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-5_acq-t1w_mtsat.nii.gz')

    # --out_prefix
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            in_mask,
                            '--in_mtoff', in_e1_mtoff, in_e2_mtoff,
                            in_e3_mtoff, in_e4_mtoff, in_e5_mtoff,
                            '--in_mton', in_e1_mton, in_e2_mton, in_e3_mton,
                            in_e4_mton, in_e5_mton,
                            '--in_t1w', in_e1_t1w, in_e2_t1w, in_e3_t1w,
                            in_e4_t1w, in_e5_t1w,
                            '--out_prefix', 'sub_01',
                            '-f')
    assert ret.success


def test_execution_MT_B1_map(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    in_mask = os.path.join(get_home(), 'MT', 'mask.nii.gz')

    in_e1_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-1_acq-mtoff_mtsat.nii.gz')
    in_e2_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-2_acq-mtoff_mtsat.nii.gz')
    in_e3_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-3_acq-mtoff_mtsat.nii.gz')
    in_e4_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-4_acq-mtoff_mtsat.nii.gz')
    in_e5_mtoff = os.path.join(get_home(),
                               'MT', 'sub-001_echo-5_acq-mtoff_mtsat.nii.gz')

    in_e1_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-1_acq-mton_mtsat.nii.gz')
    in_e2_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-2_acq-mton_mtsat.nii.gz')
    in_e3_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-3_acq-mton_mtsat.nii.gz')
    in_e4_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-4_acq-mton_mtsat.nii.gz')
    in_e5_mton = os.path.join(get_home(),
                              'MT', 'sub-001_echo-5_acq-mton_mtsat.nii.gz')

    in_e1_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-1_acq-t1w_mtsat.nii.gz')
    in_e2_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-2_acq-t1w_mtsat.nii.gz')
    in_e3_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-3_acq-t1w_mtsat.nii.gz')
    in_e4_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-4_acq-t1w_mtsat.nii.gz')
    in_e5_t1w = os.path.join(get_home(),
                             'MT', 'sub-001_echo-5_acq-t1w_mtsat.nii.gz')
    in_b1_map = os.path.join(get_home(),
                             'MT', 'sub-001_run-01_B1map.nii.gz')

    # --in_B1_map
    ret = script_runner.run('scil_mti_maps_MT.py', tmp_dir.name,
                            in_mask,
                            '--in_mtoff', in_e1_mtoff, in_e2_mtoff,
                            in_e3_mtoff, in_e4_mtoff, in_e5_mtoff,
                            '--in_mton', in_e1_mton, in_e2_mton, in_e3_mton,
                            in_e4_mton, in_e5_mton,
                            '--in_t1w', in_e1_t1w, in_e2_t1w, in_e3_t1w,
                            in_e4_t1w, in_e5_t1w,
                            '--in_B1_map', in_b1_map,
                            '--out_prefix', 'sub-01',
                            '-f')
    assert ret.success
