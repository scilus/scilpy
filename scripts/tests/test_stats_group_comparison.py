#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['stats.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(
        'scil_stats_group_comparison.py',
        '--help')
    assert ret.success


def test_execution_bundles(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_json = os.path.join(get_home(), 'stats/group', 'participants.tsv')
    in_participants = os.path.join(get_home(), 'stats/group', 'meanstd_all.json')

    ret = script_runner.run('scil_stats_group_comparison.py',
                            in_participants, in_json, 'Group',
                            '-b', 'AF_L',
                            '-m', 'FIT_FW',
                            '--va', 'mean',
                            '--gg')

    assert ret.success
