#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_denoise_sf.py',
                            '--help')
    assert ret.success


def test_execution_denoise_sf(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking', 'fodf.nii.gz')

    # We use a low resolution sphere to reduce execution time
    ret = script_runner.run('scil_denoise_sf.py', in_fodf,
                            'averaged_out.nii.gz', '--sphere', 'repulsion100')
    assert ret.success
