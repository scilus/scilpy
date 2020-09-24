#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_visualize_fodf.py', '--help')
    assert ret.success


def test_interpolation_without_background(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking',
                           'fodf.nii.gz')
    ret = script_runner.run('scil_visualize_fodf.py', in_fodf,
                            '--bg_interpolation', 'linear')
    assert (not ret.success)


def test_offset_without_background(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking',
                           'fodf.nii.gz')

    ret = script_runner.run('scil_visualize_fodf.py', in_fodf,
                            '--bg_offset', '0.5')

    assert (not ret.success)


def test_range_without_background(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking',
                           'fodf.nii.gz')

    ret = script_runner.run('scil_visualize_fodf.py', in_fodf,
                            '--bg_range', '0.0', '1.0')

    assert (not ret.success)


def test_peaks_full_basis(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking',
                           'fodf.nii.gz')
    in_peaks = os.path.join(get_home(), 'tracking',
                            'peaks.nii.gz')

    ret = script_runner.run('scil_visualize_fodf.py', in_fodf,
                            '--full_basis', '--peaks', in_peaks)

    assert (not ret.success)


def test_silent_without_output(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking',
                           'fodf.nii.gz')

    ret = script_runner.run('scil_visualize_fodf.py', in_fodf, '--silent')

    assert (not ret.success)
