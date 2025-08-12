#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_bundle_mean_fixel_afd', '--help'])
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracking = os.path.join(SCILPY_HOME, 'processing', 'tracking.trk')
    in_fodf = os.path.join(SCILPY_HOME, 'processing',
                           'fodf_descoteaux07.nii.gz')
    ret = script_runner.run(['scil_bundle_mean_fixel_afd', in_tracking,
                             in_fodf, 'afd_test.nii.gz',
                             '--sh_basis', 'descoteaux07',
                             '--length_weighting'])
    assert ret.success
