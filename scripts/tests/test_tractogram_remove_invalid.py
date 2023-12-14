#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bundles.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_remove_invalid.py', '--help')
    assert ret.success


def test_execution_bundles(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(get_home(), 'bundles',
                                 'bundle_all_1mm.trk')
    ret = script_runner.run('scil_tractogram_remove_invalid.py',
                            in_tractogram, 'bundle_all_1mm.trk', '--cut',
                            '--remove_overlapping', '--remove_single', '-f')
    assert ret.success
