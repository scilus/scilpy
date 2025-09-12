#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pytest
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


@pytest.mark.smoke
def test_help_option(script_runner):
    ret = script_runner.run(['scil_json_convert_entries_to_xlsx', '--help'])
    assert ret.success


@pytest.mark.smoke
def test_execution_tractometry(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_json = os.path.join(SCILPY_HOME, 'tractometry',
                           'length_stats_1.json')
    ret = script_runner.run(['scil_json_convert_entries_to_xlsx', in_json,
                             'length_stats.xlsx'])

    assert ret.success
