#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

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
                             '--save_as_color', '--processes', '4'])
    assert ret.success