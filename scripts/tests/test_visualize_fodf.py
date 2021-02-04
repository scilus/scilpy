#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_visualize_fodf.py', '--help')
    assert ret.success


def test_peaks_full_basis(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking',
                           'fodf.nii.gz')
    in_peaks = os.path.join(get_home(), 'tracking',
                            'peaks.nii.gz')
    # Tests that the use of a full SH basis with peaks raises a warning
    with warnings.catch_warnings(record=True) as w:
        ret = script_runner.run('scil_visualize_fodf.py', in_fodf,
                                '--full_basis', '--peaks', in_peaks)
        assert(len(w) > 0)
        assert(issubclass(w[0].category, UserWarning))
        assert('Asymmetric peaks visualization is not supported '
               'by FURY. Peaks shown as symmetric peaks.' in
               str(w[0].message))

    # The whole execution should fail because
    # the input fODF is not in full basis
    assert (not ret.success)


def test_silent_without_output(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'tracking',
                           'fodf.nii.gz')

    ret = script_runner.run('scil_visualize_fodf.py', in_fodf, '--silent')

    assert (not ret.success)
