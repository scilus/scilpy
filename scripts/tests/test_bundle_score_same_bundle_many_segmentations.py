#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bundles.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(
        'scil_bundle_score_same_bundle_many_segmentations.py', '--help')
    assert ret.success


def test_execution_bundles(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_1 = os.path.join(get_home(), 'bundles', 'bundle_0_reco.tck')
    in_2 = os.path.join(get_home(), 'bundles', 'voting_results',
                        'bundle_0.trk')
    in_ref = os.path.join(get_home(), 'bundles', 'bundle_all_1mm.nii.gz')
    in_tractogram = os.path.join(get_home(), 'bundles', 'bundle_all_1mm.trk')
    in_model = os.path.join(get_home(), 'bundles', 'fibercup_atlas',
                            'subj_1', 'bundle_0.trk')
    ret = script_runner.run(
        'scil_bundle_score_same_bundle_many_segmentations.py',
        in_1, in_2, 'AF_L_binary.json', '--streamlines_measures', in_model,
        in_tractogram, '--processes', '1', '--reference', in_ref)
    assert ret.success
