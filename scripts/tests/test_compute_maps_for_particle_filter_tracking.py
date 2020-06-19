#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_maps_for_particle_filter_tracking.py',
                            '--help')
    assert ret.success


def test_execution_tracking(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_wm = os.path.join(get_home(), 'tracking',
                            'map_wm.nii.gz')
    input_gm = os.path.join(get_home(), 'tracking',
                            'map_gm.nii.gz')
    input_csf = os.path.join(get_home(), 'tracking',
                             'map_csf.nii.gz')
    ret = script_runner.run('scil_compute_maps_for_particle_filter_tracking.py',
                            input_wm, input_gm, input_csf)
    assert ret.success
