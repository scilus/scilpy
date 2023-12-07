#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['filtering.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_detect_loops.py', '--help')
    assert ret.success


def test_execution_filtering(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'filtering',
                             'bundle_4_filtered.trk')
    ret = script_runner.run('scil_tractogram_detect_loops.py',
                            in_bundle, 'bundle_4_filtered_no_loops.trk',
                            '--looping_tractogram',
                            'bundle_4_filtered_loops.trk',
                            '-a', '270', '--qb', '--threshold', '4',
                            '--processes', '1')
    assert ret.success
