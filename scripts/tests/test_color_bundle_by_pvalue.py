#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import numpy as np

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home

fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_color_bundle_by_pvalue.py', '--help')
    assert ret.success


def test_execution_base(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    # dummy p-values file
    pvalues = np.linspace(0.0, 1.0, 100)
    np.savetxt('pvalues.txt', pvalues, delimiter=' ')

    in_bundle = os.path.join(get_home(), 'tractometry',
                             'IFGWM.trk')

    ret = script_runner.run('scil_color_bundle_by_pvalue.py',
                            in_bundle, 'pvalues.txt',
                            'out.trk', 'out.png', '--colormap', 'plasma')
    assert ret.success


def test_execution_resample(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    # dummy p-values file
    pvalues = np.linspace(0.0, 1.0, 100)
    np.savetxt('pvalues.txt', pvalues, delimiter=' ')

    in_bundle = os.path.join(get_home(), 'tractometry',
                             'IFGWM.trk')

    ret = script_runner.run('scil_color_bundle_by_pvalue.py',
                            in_bundle, 'pvalues.txt',
                            'out.trk', 'out.png', '--colormap', 'plasma',
                            '--resample', '8', '-f')
    assert ret.success
