#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
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
                      in_bundle, t1_on_bundle,
                      '--in_maps', in_t1,
                      '--out_dpp_name', 't1')

    ret = script_runner.run('scil_tractogram_point_math.py',
                            'mean',
                            'dps',
                            t1_on_bundle,
                            't1_mean_on_streamlines.trk',
                            '--in_dpp_name', 't1',
                            '--out_name', 't1_mean')

    assert ret.success


def test_execution_tractogram_point_math_mean_4D_correlation(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'tracking',
                             'local_split_0.trk')

    in_fodf = os.path.join(get_home(), 'tracking',
                           'fodf.nii.gz')
    fodf_on_bundle = 'fodf_on_streamlines.trk'

    script_runner.run('scil_tractogram_project_map_to_streamlines.py',
                      in_bundle, fodf_on_bundle,
                      '--in_maps', in_fodf, in_fodf,
                      '--out_dpp_name', 'fodf', 'fodf2')

    ret = script_runner.run('scil_tractogram_point_math.py',
                            'correlation',
                            'dps',
                            fodf_on_bundle,
                            'fodf_correlation_on_streamlines.trk',
                            '--in_dpp_name', 'fodf',
                            '--out_name', 'fodf_correlation')

    assert ret.success
