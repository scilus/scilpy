#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['processing.zip'])


def test_help_option(script_runner):
    ret = script_runner.run(['scil_viz_tractogram_seeds', '--help'])
    assert ret.success


def test_run_option(script_runner):
    seed_map = os.path.join(SCILPY_HOME, 'processing', 'fa.nii.gz')

    # To test option --tractogram, we would need a tractogram with seeds saved.
    ret = script_runner.run(['scil_viz_tractogram_seeds_3d', seed_map,
                            '--silent'])
    assert ret.success
