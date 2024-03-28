#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_json_merge_entries.py', '--help')
    assert ret.success


def test_execution_tractometry(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_json_1 = os.path.join(SCILPY_HOME, 'tractometry',
                           'length_stats_1.json')
    in_json_2 = os.path.join(SCILPY_HOME, 'tractometry',
                           'length_stats_2.json')
    ret = script_runner.run('scil_json_merge_entries.py', in_json_1,
                            in_json_2, 'merge.json', '--keep_separate')

    assert ret.success
