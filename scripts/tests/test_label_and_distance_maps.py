#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_resample_streamlines.py', '--help')
    assert ret.success


def test_execution_tractometry(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_bundle = os.path.join(get_home(), 'tractometry',
                                'IFGWM.trk')
    input_centroid = os.path.join(get_home(), 'tractometry',
                                  'IFGWM_uni_c_10.trk')
    ret = script_runner.run('scil_label_and_distance_maps.py',
                            input_bundle, input_centroid,
                            'label.npz', 'distance.npz')
    assert ret.success
