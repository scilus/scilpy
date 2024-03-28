#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bundles.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_segment_bundles.py', '--help')
    assert ret.success


def test_execution_bundles(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'bundles',
                                 'bundle_all_1mm.trk')
    in_models = os.path.join(SCILPY_HOME, 'bundles', 'fibercup_atlas')
    in_aff = os.path.join(SCILPY_HOME, 'bundles',
                          'affine.txt')

    tmp_config = {}
    for i in range(1, 6):
        tmp_config['bundle_{}.trk'.format(i)] = 4

    with open('config.json', 'w') as outfile:
        json.dump(tmp_config, outfile)

    ret = script_runner.run('scil_tractogram_segment_bundles.py',
                            in_tractogram, 'config.json',
                            in_models,
                            in_aff, '--inverse',
                            '--processes', '1', '-v', 'WARNING')
    assert ret.success
