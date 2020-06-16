#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bundles.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_evaluate_bundles_binary_classification_measures.py', '--help')
    assert ret.success


def test_execution_bundles(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_tractogram = os.path.join(get_home(), 'bundles',
                                    'bundle_all_1mm.trk')
    input_ref = os.path.join(get_home(), 'bundles',
                             'avg_dwi.nii.gz')
    input_model = os.path.join(get_home(), 'bundles', 'fake_atlas',
                               'subj_1', 'bundle_0.tck')
    ret = script_runner.run('scil_evaluate_bundles_binary_classification_measures.py',
                            'bundle_0_reco.tck', 'voting_results/bundle_0.trk',
                            'AF_L_binary.json', '--streamlines_measures',
                            input_model, input_tractogram, '--processes', '1',
                            '--reference', input_ref)
    assert ret.success
