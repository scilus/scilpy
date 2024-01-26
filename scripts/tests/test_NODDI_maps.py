#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_NODDI_maps.py', '--help')
    assert ret.success


def test_execution_commit_amico(script_runner, mock_collector):
    os.chdir(os.path.expanduser(tmp_dir.name))
    _mocks = mock_collector(["amico_evaluator"])

    in_dwi = os.path.join(get_home(), 'commit_amico', 'dwi.nii.gz')
    in_bval = os.path.join(get_home(), 'commit_amico', 'dwi.bval')
    in_bvec = os.path.join(get_home(), 'commit_amico', 'dwi.bvec')
    mask = os.path.join(get_home(), 'commit_amico', 'mask.nii.gz')

    ret = script_runner.run('scil_NODDI_maps.py', in_dwi,
                            in_bval, in_bvec, '--mask', mask,
                            '--out_dir', 'noddi', '--b_thr', '30',
                            '--para_diff', '0.0017', '--iso_diff', '0.003',
                            '--lambda1', '0.5', '--lambda2', '0.001',
                            '--processes', '1')

    assert ret.success

    if _mocks["amico_evaluator"]:
        amico_mock = _mocks["amico_evaluator"]
        assert amico_mock["fit"].called_once()
        assert amico_mock["generate_kernels"].called_once()
        assert amico_mock["save_results"].called_once()
