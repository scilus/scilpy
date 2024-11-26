#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(
        'scil_connectivity_compute_simple_matrix.py', '--help')
    assert ret.success


def test_script(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_labels = os.path.join(SCILPY_HOME, 'tractometry',
                             'IFGWM_labels_map.nii.gz')
    in_sft = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM.trk')

    ret = script_runner.run(
        'scil_connectivity_compute_simple_matrix.py', in_sft, in_labels,
        'out_matrix.npy', 'out_labels.txt', '--hide_labels', '10',
        '--percentage', '--hide_fig', '--out_fig', 'matrices.png')
    assert ret.success
