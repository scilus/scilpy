#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_extract_ushape.py',
                            '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_trk = os.path.join(get_home(), 'tracking', 'union.trk')
    out_trk = 'ushape.trk'
    remaining_trk = 'remaining.trk'
    ret = script_runner.run('scil_tractogram_extract_ushape.py', in_trk, out_trk,
                            '--minU', '0.5',
                            '--maxU', '1',
                            '--remaining_tractogram', remaining_trk,
                            '--display_counts')
    assert ret.success
