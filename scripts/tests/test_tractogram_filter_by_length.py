#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.io.streamlines import load_tractogram

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['filtering.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_filter_by_length.py',
                            '--help')
    assert ret.success


def test_execution_filtering(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    # Effectively, this doesn't filter anything, since bundle_4.trk has
    # all streamlines with lengths of 130mm. This is just to test the
    # script execution.
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    ret = script_runner.run('scil_tractogram_filter_by_length.py',
                            in_bundle,  'bundle_4_filtered.trk',
                            '--minL', '125', '--maxL', '130')

    sft = load_tractogram('bundle_4_filtered.trk', 'same')
    assert len(sft) == 52

    assert ret.success


def test_rejected_filtering(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_all_1mm.trk')
    ret = script_runner.run('scil_tractogram_filter_by_length.py',
                            in_bundle,  'bundle_all_1mm_filtered.trk',
                            '--minL', '125', '--maxL', '130',
                            '--out_rejected', 'bundle_all_1mm_rejected.trk')
    assert ret.success
    assert os.path.exists('bundle_all_1mm_rejected.trk')
    assert os.path.exists('bundle_all_1mm_rejected.trk')

    sft = load_tractogram('bundle_all_1mm_filtered.trk', 'same')
    rejected_sft = load_tractogram('bundle_all_1mm_rejected.trk', 'same')

    assert len(sft) == 266
    assert len(rejected_sft) == 2824


def test_rejected_filtering_no_rejection(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_4.trk')
    ret = script_runner.run('scil_tractogram_filter_by_length.py',
                            in_bundle,  'bundle_4_filtered_no_rejection.trk',
                            '--minL', '125', '--maxL', '130',
                            '--out_rejected', 'bundle_4_rejected.trk')
    assert ret.success

    # File should be created even though there are no rejected streamlines
    assert os.path.exists('bundle_4_rejected.trk')

    sft = load_tractogram('bundle_4_filtered_no_rejection.trk', 'same')
    rejected_sft = load_tractogram('bundle_4_rejected.trk', 'same')

    assert len(sft) == 52
    assert len(rejected_sft) == 0
