#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bst.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_screenshot(script_runner):
    in_fa = os.path.join(SCILPY_HOME, 'bst', 'fa.nii.gz')

    ret = script_runner.run("scil_viz_volume_screenshot.py", in_fa, 'fa.png')
    assert ret.success


def test_help_option(script_runner):
    ret = script_runner.run("scil_viz_volume_screenshot.py", "--help")
    assert ret.success
