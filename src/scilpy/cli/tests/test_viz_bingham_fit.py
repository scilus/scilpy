#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_viz_bingham_fit', '--help'])
    assert ret.success


def test_silent_without_output(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_dummy = os.path.join(SCILPY_HOME, 'processing', 'fodf_bingham.nii.gz')
    out = os.path.join(tmp_dir.name, 'test_bingham.png')
    ret = script_runner.run(['scil_viz_bingham_fit', in_dummy,
                             '--silent', '--output', out])

    assert ret.success
