#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import get_testing_files_dict, fetch_data


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bundles.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(
        'scil_tractogram_pairwise_comparison.py', '--help')
    assert ret.success


def test_execution_bundles_skip(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_1 = os.path.join(SCILPY_HOME, 'bundles', 'bundle_0_reco.tck')
    in_2 = os.path.join(SCILPY_HOME, 'bundles', 'voting_results',
                        'bundle_0.trk')
    in_ref = os.path.join(SCILPY_HOME, 'bundles', 'bundle_all_1mm.nii.gz')
    ret = script_runner.run('scil_tractogram_pairwise_comparison.py',
        in_1, in_2, '--out_dir', tmp_dir.name, '--reference', in_ref,
        '--skip_streamlines_distance')
    assert ret.success

def test_execution_bundles(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_1 = os.path.join(SCILPY_HOME, 'bundles', 'bundle_0_reco.tck')
    in_2 = os.path.join(SCILPY_HOME, 'bundles', 'voting_results',
                        'bundle_0.trk')
    in_ref = os.path.join(SCILPY_HOME, 'bundles', 'bundle_all_1mm.nii.gz')
    ret = script_runner.run('scil_tractogram_pairwise_comparison.py',
        in_1, in_2, '--out_dir', tmp_dir.name, '--reference', in_ref, '-f',
        '--skip_streamlines_distance')
    assert ret.success