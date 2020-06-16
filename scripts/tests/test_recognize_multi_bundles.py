#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bundles.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_recognize_multi_bundles.py', '--help')
    assert ret.success


def test_execution_bundles(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_tractogram = os.path.join(get_home(), 'bundles',
                                    'bundle_all_1mm.trk')
    input_conf = os.path.join(get_home(), 'bundles', 'fake_atlas',
                              'default_config_sim.json')
    input_model_1 = os.path.join(get_home(), 'bundles', 'fake_atlas',
                                 'subj_1/')
    input_model_2 = os.path.join(get_home(), 'bundles', 'fake_atlas',
                                 'subj_2/')
    input_model_3 = os.path.join(get_home(), 'bundles', 'fake_atlas',
                                 'subj_3/')
    input_aff = os.path.join(get_home(), 'bundles',
                             'affine.txt')
    ret = script_runner.run('scil_recognize_multi_bundles.py', input_tractogram,
                            input_conf, input_model_1, input_model_2,
                            input_model_3, input_aff, '--inverse',
                            '--tractogram_clustering_thr', '15',
                            '--processes', '1', '--log', 'WARNING')
    assert ret.success