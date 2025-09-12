#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pytest
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bst.zip'])
tmp_dir = tempfile.TemporaryDirectory()


@pytest.mark.smoke
def test_help_option(script_runner):
    ret = script_runner.run(['scil_viz_volume_screenshot_mosaic', '--help'])
    assert ret.success


@pytest.mark.smoke
def test_screenshot(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fa = os.path.join(SCILPY_HOME, 'bst', 'fa.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'bst', 'mask.nii.gz')

    ret = script_runner.run(["scil_viz_volume_screenshot_mosaic", '1', '1',
                            in_fa, in_mask, 'fa.png', '50'])
    assert ret.success
