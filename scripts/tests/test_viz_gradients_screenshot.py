#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_viz_gradients_screenshot.py', '--help'])
    assert ret.success


def test_run_bval(script_runner):
    bval = os.path.join(SCILPY_HOME, 'processing', 'dwi.bval')
    bvec = os.path.join(SCILPY_HOME, 'processing', 'dwi.bvec')
    ret = script_runner.run('scil_viz_gradients_screenshot.py',
                            '--in_gradient_scheme', bval, bvec,
                            '--test_run')
    assert ret.success


def test_run_dipy_sphere(script_runner):
    ret = script_runner.run('scil_viz_gradients_screenshot.py',
                            '--dipy_sphere', 'symmetric362',
                            '--test_run')
    assert ret.success