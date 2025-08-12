#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_viz_fodf', '--help'])
    assert ret.success


def test_silent_without_output(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz')

    ret = script_runner.run(['scil_viz_fodf', in_fodf, '--silent'])

    # Should say that requires an output with --silent mode
    assert (not ret.success)


def test_run(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'processing', 'seed.nii.gz')
    # No variance file in our test data, but faking it with the fodf file.
    in_variance = os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz')
    out_name = os.path.join(tmp_dir.name, 'out.png')
    ret = script_runner.run(['scil_viz_fodf', in_fodf, '--silent',
                             '--in_transparency_mask', in_mask,
                             '--mask', in_mask,
                             '--variance', in_variance,
                             '--output', out_name])
    assert ret.success


def test_run_sphsubdivide(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'processing', 'seed.nii.gz')
    out_name = os.path.join(tmp_dir.name, 'out2.png')

    # Note. Cannot add --sph_subdivide to the test above, causes a memory
    # crash. Without the variance, lighter.
    ret = script_runner.run(['scil_viz_fodf', in_fodf, '--silent',
                             '--mask', in_mask,
                             '--sph_subdivide', '2',
                             '--sphere', 'repulsion100',
                             '--output', out_name])

    assert ret.success
