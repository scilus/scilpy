#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_shuffle.py', '--help')
    assert ret.success


def test_execution_tracking(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto = os.path.join(SCILPY_HOME, 'tracking', 'union.trk')
    ret = script_runner.run('scil_tractogram_shuffle.py', in_tracto,
                            'union_shuffle.trk')
    assert ret.success
