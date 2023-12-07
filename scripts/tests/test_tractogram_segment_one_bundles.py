#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bundles.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_segment_one_bundles.py', '--help')
    assert ret.success


def test_execution_bundles(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(get_home(), 'bundles',
                                 'bundle_all_1mm.trk')
    in_model = os.path.join(get_home(), 'bundles', 'fibercup_atlas',
                            'subj_1', 'bundle_0.trk')
    in_aff = os.path.join(get_home(), 'bundles',
                          'affine.txt')
    in_ref = os.path.join(get_home(), 'bundles',
                          'bundle_all_1mm.nii.gz')
    ret = script_runner.run('scil_tractogram_segment_one_bundles.py',
                            in_tractogram, in_model, in_aff,
                            'bundle_0_reco.tck', '--inverse',
                            '--tractogram_clustering_thr', '12',
                            '--slr_threads', '1', '--out_pickle',
                            'clusters.pkl', '--reference', in_ref)
    assert ret.success
