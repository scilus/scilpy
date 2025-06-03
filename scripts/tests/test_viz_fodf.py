#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_viz_fodf.py', '--help')
    assert ret.success


def test_silent_without_output(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')

    ret = script_runner.run('scil_viz_fodf.py', in_fodf, '--silent')

    # Should say that requires an output with --silent mode
    assert (not ret.success)


def test_run(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')
    out_name = os.path.join(tmp_dir.name, 'out.png')
    ret = script_runner.run('scil_viz_fodf.py', in_fodf, '--silent',
                            '--in_transparency_mask', in_mask,
                            '--output', out_name)

    assert ret.success
