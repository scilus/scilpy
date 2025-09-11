#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_gradients_normalize_bvecs',
                            '--help'])
    assert ret.success


def test_execution_processing_fsl(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')
    ret = script_runner.run(['scil_gradients_normalize_bvecs',
                            in_bvec, '1000_norm.bvec'])
    assert ret.success


def test_normalization_axis_is_right(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    bvec = [
        [1., 2., 3., 4., 5., 6.],
        [1., 1., 1., 1., 1., 1.],
        [1., 1., 1., 1., 1., 1.]
    ]

    expected_norms = [np.sqrt(3.), np.sqrt(6.), np.sqrt(11.),
                      np.sqrt(18.), np.sqrt(27.), np.sqrt(38.)]
    expected_bvecs = np.asarray(bvec) / expected_norms

    in_bvec = os.path.join(tmp_dir.name, 'in.bvec')
    out_bvec = os.path.join(tmp_dir.name, 'expected.bvec')
    np.savetxt(in_bvec, bvec, fmt='%.8f')
    ret = script_runner.run(['scil_gradients_normalize_bvecs',
                             in_bvec, out_bvec])
    assert ret.success

    out_bvec = np.loadtxt(out_bvec)
    np.testing.assert_almost_equal(out_bvec, expected_bvecs, decimal=6)
    np.testing.assert_almost_equal(np.linalg.norm(out_bvec, axis=0),
                                   np.ones(len(bvec[0])), decimal=6)
