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
    ret = script_runner.run(['scil_json_harmonize_entries', '--help'])
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_json = os.path.join(SCILPY_HOME, 'tractometry',
                           'metric_label.json')
    ret = script_runner.run(['scil_json_harmonize_entries', in_json,
                             'tmp.json', '--indent', '3',
                             '--sort_keys', '-f'])
    assert ret.success
