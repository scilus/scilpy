#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import get_testing_files_dict, fetch_data


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_bundle_label_map.py', '--help')
    assert ret.success


def test_execution_tractometry_euclidian(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry',
                             'IFGWM.trk')
    in_centroid = os.path.join(SCILPY_HOME, 'tractometry',
                               'IFGWM_uni_c_10.trk')
    ret = script_runner.run('scil_bundle_label_map.py',
                            in_bundle, in_centroid,
                            'results_euc/',
                            '--colormap', 'viridis')
    assert ret.success


def test_execution_tractometry_hyperplane(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'tractometry',
                             'IFGWM.trk')
    in_centroid = os.path.join(SCILPY_HOME, 'tractometry',
                               'IFGWM_uni_c_10.trk')
    ret = script_runner.run('scil_bundle_label_map.py',
                            in_bundle, in_centroid,
                            'results_man/',
                            '--colormap', 'viridis',
                            '--hyperplane', '--use_manhattan')
    assert ret.success
