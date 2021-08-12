#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_tracking_branching.py',
                            '--help')
    assert ret.success


def test_execution_tracking_fodf(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking',
                           'fodf.nii.gz')
    in_mask = os.path.join(get_home(), 'tracking',
                           'seeding_mask.nii.gz')
    ret = script_runner.run('scil_compute_tracking_branching.py', in_fodf,
                            in_mask, in_mask, 'local_micro.trk', '--nt', '100')
    assert ret.success

