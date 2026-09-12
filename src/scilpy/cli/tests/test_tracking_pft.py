#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_tracking_pft',
                            '--help'])
    assert ret.success


def test_execution_tracking(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking',
                           'fodf.nii.gz')
    in_interface = os.path.join(SCILPY_HOME, 'tracking',
                                'interface.nii.gz')
    in_include = os.path.join(SCILPY_HOME, 'tracking',
                              'map_include.nii.gz')
    in_exclude = os.path.join(SCILPY_HOME, 'tracking',
                              'map_exclude.nii.gz')
    ret = script_runner.run(['scil_tracking_pft', in_fodf,
                             in_interface, in_include, in_exclude,
                             'pft.trk', '--nt', '1000', '--compress', '0.1',
                             '--sh_basis', 'descoteaux07',
                             '--min_length', '20',
                             '--max_length', '200'])
    assert ret.success


def test_execution_tracking_pft_sf_rel_thr(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_interface = os.path.join(SCILPY_HOME, 'tracking', 'interface.nii.gz')
    in_include = os.path.join(SCILPY_HOME, 'tracking', 'map_include.nii.gz')
    in_exclude = os.path.join(SCILPY_HOME, 'tracking', 'map_exclude.nii.gz')

    ret = script_runner.run(['scil_tracking_pft', in_fodf,
                             in_interface, in_include, in_exclude,
                             'pft_sf_rel.trk', '--nt', '10',
                             '--sh_basis', 'descoteaux07',
                             '--global_sf_rel_thr', '0.1'])
    assert ret.success


def test_execution_tracking_pft_sf_abs_thr(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_interface = os.path.join(SCILPY_HOME, 'tracking', 'interface.nii.gz')
    in_include = os.path.join(SCILPY_HOME, 'tracking', 'map_include.nii.gz')
    in_exclude = os.path.join(SCILPY_HOME, 'tracking', 'map_exclude.nii.gz')

    ret = script_runner.run(['scil_tracking_pft', in_fodf,
                             in_interface, in_include, in_exclude,
                             'pft_sf_abs.trk', '--nt', '10',
                             '--sh_basis', 'descoteaux07',
                             '--global_sf_abs_thr', '0.01'])
    assert ret.success
