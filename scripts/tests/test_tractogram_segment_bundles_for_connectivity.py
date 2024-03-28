#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(
        'scil_tractogram_segment_bundles_for_connectivity.py', '--help')
    assert ret.success


def test_execution_connectivity(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'connectivity', 'bundle_all_1mm.trk')
    in_atlas = os.path.join(SCILPY_HOME, 'connectivity',
                            'endpoints_atlas.nii.gz')
    ret = script_runner.run(
        'scil_tractogram_segment_bundles_for_connectivity.py', in_bundle,
        in_atlas, 'decompose.h5', '--min_length', '20', '--max_length', '200',
        '--outlier_threshold', '0.5', '--loop_max_angle', '330',
        '--curv_qb_distance', '10', '--processes', '1')
    assert ret.success
