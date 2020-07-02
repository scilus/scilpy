#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_convert_json_to_xlsx.py', '--help')
    assert ret.success


def test_execution_tractometry(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_json = os.path.join(get_home(), 'tractometry',
                           'length_stats_1.json')
    ret = script_runner.run('scil_convert_json_to_xlsx.py', in_json,
                            'length_stats.xlsx')

    assert ret.success
