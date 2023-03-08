#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()


##### Deprecated file but it should still be running.
##### For more exhaustive tests, see test_tractogram_math.py

def test_help_option(script_runner):
    ret = script_runner.run('scil_streamlines_math.py', '--help')
    assert ret.success


def test_execution_union_color(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tracto_1 = os.path.join(get_home(), 'others',
                               'fibercup_bundles_color.trk')
    in_tracto_2 = os.path.join(get_home(), 'others',
                               'fibercup_bundle_0_color.trk')
    ret = script_runner.run('scil_streamlines_math.py', 'union',
                            in_tracto_1, in_tracto_2, 'union_color.trk')
    assert ret.success
