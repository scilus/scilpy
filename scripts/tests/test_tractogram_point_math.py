#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_point_math.py', '--help')
    assert ret.success


def test_execution_tractogram_point_math_mean_3D_defaults(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'tractometry',
                             'IFGWM_uni.trk')
    in_t1 = os.path.join(get_home(), 'tractometry',
                         'mni_masked.nii.gz')

    t1_on_bundle = 't1_on_streamlines.trk'

    script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                      in_bundle, in_t1, t1_on_bundle)

    ret = script_runner.run('scil_tractogram_point_math.py', 'mean', t1_on_bundle,
                            't1_mean_on_streamlines.trk')

    assert ret.success
