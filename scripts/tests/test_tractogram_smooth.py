#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_smooth.py', '--help')
    assert ret.success


def test_execution_tracking(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto = os.path.join(get_home(), 'tracking',
                             'union_shuffle_sub.trk')
    ret = script_runner.run('scil_tractogram_smooth.py', in_tracto,
                            'union_shuffle_sub_smooth.trk', '--gaussian', '10',
                            '-e', '0.05')
    assert ret.success
