#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tracking_local_dev.py',
                            '--help')
    assert ret.success


def test_execution_tracking_fodf(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking',
                           'fodf.nii.gz')
    in_mask = os.path.join(get_home(), 'tracking',
                           'seeding_mask.nii.gz')
    ret = script_runner.run('scil_tracking_local_dev.py', in_fodf,
                            in_mask, in_mask, 'local_prob.trk', '--nt', '10',
                            '--compress', '0.1', '--sh_basis', 'descoteaux07',
                            '--min_length', '20', '--max_length', '200',
                            '--save_seeds', '--rng_seed', '0')
    assert ret.success
