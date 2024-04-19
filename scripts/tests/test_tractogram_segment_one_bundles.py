#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bundles.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_segment_one_bundle.py', '--help')
    assert ret.success


def test_execution_bundles(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'bundles',
                                 'bundle_all_1mm.trk')
    in_model = os.path.join(SCILPY_HOME, 'bundles', 'fibercup_atlas',
                            'subj_1', 'bundle_0.trk')
    in_aff = os.path.join(SCILPY_HOME, 'bundles',
                          'affine.txt')
    in_ref = os.path.join(SCILPY_HOME, 'bundles',
                          'bundle_all_1mm.nii.gz')
    ret = script_runner.run('scil_tractogram_segment_one_bundle.py',
                            in_tractogram, in_model, in_aff,
                            'bundle_0_reco.tck', '--inverse',
                            '--tractogram_clustering_thr', '12',
                            '--slr_threads', '1', '--out_pickle',
                            'clusters.pkl', '--reference', in_ref)
    assert ret.success
