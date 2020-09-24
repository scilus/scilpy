#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_visualize_scatterplot.py', '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_x = os.path.join(get_home(), 'processing',
                        'fa.nii.gz')
    in_y = os.path.join(get_home(), 'processing',
                        'ad.nii.gz')
    ret = script_runner.run('scil_visualize_scatterplot.py', in_x, in_y,
                            'scatter_plot.png')
    assert ret.success


def test_execution_processing_mask(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_x = os.path.join(get_home(), 'processing',
                        'fa.nii.gz')
    in_y = os.path.join(get_home(), 'processing',
                        'ad.nii.gz')
    in_mask = os.path.join(get_home(), 'processing',
                           'seed.nii.gz')
    ret = script_runner.run('scil_visualize_scatterplot.py', in_x, in_y,
                            'scatter_plot.png', '--in_mask', in_mask)
    assert ret.success
