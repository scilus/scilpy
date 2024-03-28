#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['stats.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(
        'scil_stats_group_comparison.py',
        '--help')
    assert ret.success


def test_execution_bundles(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_json = os.path.join(SCILPY_HOME, 'stats/group', 'participants.tsv')
    in_participants = os.path.join(SCILPY_HOME, 'stats/group',
                                   'meanstd_all.json')

    ret = script_runner.run('scil_stats_group_comparison.py',
                            in_participants, in_json, 'Group',
                            '-b', 'AF_L',
                            '-m', 'FIT_FW',
                            '--va', 'mean',
                            '--gg')

    assert ret.success
