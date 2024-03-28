#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['filtering.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_bundle_filter_by_occurence.py', '--help')
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_1 = os.path.join(SCILPY_HOME, 'filtering', 'bundle_4.trk')
    in_2 = os.path.join(SCILPY_HOME, 'filtering', 'bundle_4_filtered.trk')
    in_3 = os.path.join(SCILPY_HOME, 'filtering',
                        'bundle_4_filtered_no_loops.trk')

    prefix = 'test_voting_'
    ret = script_runner.run('scil_bundle_filter_by_occurence.py', in_1, in_2,
                            in_3, prefix, '--ratio_streamlines', '0.5',
                            '--ratio_voxels', '0.5')
    assert ret.success
