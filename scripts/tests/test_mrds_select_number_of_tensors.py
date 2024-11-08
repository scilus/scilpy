#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['mrds.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_mrds_select_number_of_tensors.py', '--help')
    assert ret.success


def test_execution_mrds(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_nufo = os.path.join(SCILPY_HOME,
                           'mrds', 'sub-01_nufo.nii.gz')
    in_v1_signal_fraction = os.path.join(SCILPY_HOME,
                                         'mrds',
                                         'sub-01_V1_signal_fraction.nii.gz')
    in_v1_evals = os.path.join(SCILPY_HOME,
                               'mrds', 'sub-01_V1_evecs.nii.gz')
    in_v1_isotropic = os.path.join(SCILPY_HOME,
                                   'mrds',
                                   'sub-01_V1_isotropic.nii.gz')
    in_v1_num_tensors = os.path.join(SCILPY_HOME,
                                     'mrds',
                                     'sub-01_V1_num_tensors.nii.gz')
    in_v1_evecs = os.path.join(SCILPY_HOME,
                               'mrds',
                               'sub-01_V1_evecs.nii.gz')
    in_v2_signal_fraction = os.path.join(SCILPY_HOME,
                                         'mrds',
                                         'sub-01_V2_signal_fraction.nii.gz')
    in_v2_evals = os.path.join(SCILPY_HOME, 'mrds',
                               'sub-01_V2_evecs.nii.gz')
    in_v2_isotropic = os.path.join(SCILPY_HOME, 'mrds',
                                   'sub-01_V2_isotropic.nii.gz')
    in_v2_num_tensors = os.path.join(SCILPY_HOME, 'mrds',
                                     'sub-01_V2_num_tensors.nii.gz')
    in_v2_evecs = os.path.join(SCILPY_HOME, 'mrds',
                               'sub-01_V2_evecs.nii.gz')
    in_v3_signal_fraction = os.path.join(SCILPY_HOME, 'mrds',
                                         'sub-01_V3_signal_fraction.nii.gz')
    in_v3_evals = os.path.join(SCILPY_HOME,
                               'mrds', 'sub-01_V3_evecs.nii.gz')
    in_v3_isotropic = os.path.join(SCILPY_HOME, 'mrds',
                                   'sub-01_V3_isotropic.nii.gz')
    in_v3_num_tensors = os.path.join(SCILPY_HOME, 'mrds',
                                     'sub-01_V3_num_tensors.nii.gz')
    in_v3_evecs = os.path.join(SCILPY_HOME, 'mrds',
                               'sub-01_V3_evecs.nii.gz')
    # no option
    ret = script_runner.run('scil_mrds_select_number_of_tensors.py',
                            in_nufo,
                            '--N1', in_v1_signal_fraction,
                            in_v1_evals, in_v1_isotropic,
                            in_v1_num_tensors, in_v1_evecs,
                            '--N2', in_v2_signal_fraction,
                            in_v2_evals, in_v2_isotropic,
                            in_v2_num_tensors, in_v2_evecs,
                            '--N3', in_v3_signal_fraction,
                            in_v3_evals, in_v3_isotropic,
                            in_v3_num_tensors, in_v3_evecs,
                            '-f')
    assert ret.success


def test_execution_mrds_w_mask(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_nufo = os.path.join(SCILPY_HOME,
                           'mrds', 'sub-01_nufo.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'mrds',
                           'sub-01_mask.nii.gz')
    in_v1_signal_fraction = os.path.join(SCILPY_HOME,
                                         'mrds',
                                         'sub-01_V1_signal_fraction.nii.gz')
    in_v1_evals = os.path.join(SCILPY_HOME,
                               'mrds', 'sub-01_V1_evecs.nii.gz')
    in_v1_isotropic = os.path.join(SCILPY_HOME,
                                   'mrds',
                                   'sub-01_V1_isotropic.nii.gz')
    in_v1_num_tensors = os.path.join(SCILPY_HOME,
                                     'mrds',
                                     'sub-01_V1_num_tensors.nii.gz')
    in_v1_evecs = os.path.join(SCILPY_HOME,
                               'mrds',
                               'sub-01_V1_evecs.nii.gz')
    in_v2_signal_fraction = os.path.join(SCILPY_HOME,
                                         'mrds',
                                         'sub-01_V2_signal_fraction.nii.gz')
    in_v2_evals = os.path.join(SCILPY_HOME, 'mrds',
                               'sub-01_V2_evecs.nii.gz')
    in_v2_isotropic = os.path.join(SCILPY_HOME, 'mrds',
                                   'sub-01_V2_isotropic.nii.gz')
    in_v2_num_tensors = os.path.join(SCILPY_HOME, 'mrds',
                                     'sub-01_V2_num_tensors.nii.gz')
    in_v2_evecs = os.path.join(SCILPY_HOME, 'mrds',
                               'sub-01_V2_evecs.nii.gz')
    in_v3_signal_fraction = os.path.join(SCILPY_HOME, 'mrds',
                                         'sub-01_V3_signal_fraction.nii.gz')
    in_v3_evals = os.path.join(SCILPY_HOME,
                               'mrds', 'sub-01_V3_evecs.nii.gz')
    in_v3_isotropic = os.path.join(SCILPY_HOME, 'mrds',
                                   'sub-01_V3_isotropic.nii.gz')
    in_v3_num_tensors = os.path.join(SCILPY_HOME, 'mrds',
                                     'sub-01_V3_num_tensors.nii.gz')
    in_v3_evecs = os.path.join(SCILPY_HOME, 'mrds',
                               'sub-01_V3_evecs.nii.gz')
    # no option
    ret = script_runner.run('scil_mrds_select_number_of_tensors.py',
                            in_nufo,
                            '--N1', in_v1_signal_fraction,
                            in_v1_evals, in_v1_isotropic,
                            in_v1_num_tensors, in_v1_evecs,
                            '--N2', in_v2_signal_fraction,
                            in_v2_evals, in_v2_isotropic,
                            in_v2_num_tensors, in_v2_evecs,
                            '--N3', in_v3_signal_fraction,
                            in_v3_evals, in_v3_isotropic,
                            in_v3_num_tensors, in_v3_evecs,
                            '--mask', in_mask,
                            '-f')
    assert ret.success
