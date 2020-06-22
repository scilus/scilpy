#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_streamlines_math.py', '--help')
    assert ret.success


def test_execution_union(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(get_home(), 'tracking',
                               'local.trk')
    in_tracto_2 = os.path.join(get_home(), 'tracking',
                               'pft.trk')
    ret = script_runner.run('scil_streamlines_math.py', 'union',
                            in_tracto_1, in_tracto_2, 'union.trk')
    assert ret.success


def test_execution_intersection(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(get_home(), 'tracking',
                               'union.trk')
    in_tracto_2 = os.path.join(get_home(), 'tracking',
                               'pft.trk')
    ret = script_runner.run('scil_streamlines_math.py', 'intersection',
                            in_tracto_1, in_tracto_2, 'intersection.trk')
    assert ret.success


def test_execution_difference(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(get_home(), 'tracking',
                               'union.trk')
    in_tracto_2 = os.path.join(get_home(), 'tracking',
                               'pft.trk')
    ret = script_runner.run('scil_streamlines_math.py', 'difference',
                            in_tracto_1, in_tracto_2, 'difference.trk')
    assert ret.success
