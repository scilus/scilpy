#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_project_streamlines_to_map.py',
                            '--help')
    assert ret.success


def test_execution_endpoints(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'tractometry', 'IFGWM_uni.trk')
    in_ref = os.path.join(get_home(), 'tractometry', 'mni_masked.nii.gz')
    out_bundle = os.path.join(get_home(), 'tractometry',
                              'IFGWM_uni_with_dpp.trk')

    # Add metrics as dpp. Or get a streamline that already as some dpp in the
    # test data.
    script_runner.run('scil_tractogram_project_map_to_streamlines', in_bundle,
                      in_ref, out_bundle, '--dpp_name', 'some_metric')

    ret = script_runner.run('scil_tractogram_project_streamlines_to_map.py',
                            out_bundle, 'out_test/endpoints_',
                            '--mean_endpoints', '--to_endpoints')
    assert ret.success

    ret = script_runner.run('scil_tractogram_project_streamlines_to_map.py',
                            out_bundle, 'out_test/endpoints_',
                            '--mean_streamline', '--to_endpoints')
    assert ret.success

    ret = script_runner.run('scil_tractogram_project_streamlines_to_map.py',
                            out_bundle, 'out_test/endpoints_',
                            '--point_by_point', '--to_wm')
    assert ret.success