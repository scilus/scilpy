#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from scilpy import SCILPY_HOME


def test_help_option(script_runner):
    ret = script_runner.run('scil_volume_stats_in_labels.py', '--help')
    assert ret.success


def test_execution(script_runner):
    in_map = os.path.join(SCILPY_HOME, 'plot', 'fa.nii.gz')
    in_atlas = os.path.join(SCILPY_HOME, 'plot', 'atlas_brainnetome.nii.gz')
    atlas_lut = os.path.join(SCILPY_HOME, 'plot', 'atlas_brainnetome.json')
    ret = script_runner.run('scil_volume_stats_in_labels.py',
                            in_atlas, atlas_lut, in_map)
    assert ret.success
