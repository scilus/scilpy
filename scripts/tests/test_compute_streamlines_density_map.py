#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['filtering.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(
        'scil_compute_streamlines_density_map.py', '--help')
    assert ret.success


def test_execution_filtering(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_bundle = os.path.join(get_home(), 'filtering',
                                'bundle_4_filtered.trk')
    ret = script_runner.run('scil_compute_streamlines_density_map.py',
                            input_bundle, 'bundle_4_filtered.nii.gz',
                            '--binary')
    assert ret.success
