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
    ret = script_runner.run('scil_tractogram_split.py', '--help')
    assert ret.success


def test_execution_tracking(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto = os.path.join(SCILPY_HOME, 'tracking',
                             'local.trk')
    ret = script_runner.run('scil_tractogram_split.py', in_tracto,
                            'local_split', '--nb_chunks', '3', '-f')
    assert ret.success

    ret = script_runner.run('scil_tractogram_split.py', in_tracto,
                            'local_split', '--nb_chunks', '3', '-f',
                            '--split_per_cluster')
    assert ret.success

    ret = script_runner.run('scil_tractogram_split.py', in_tracto,
                            'local_split', '--nb_chunks', '3', '-f',
                            '--do_not_randomize')
    assert ret.success
