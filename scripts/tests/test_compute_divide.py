#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_divide.py', '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'commit_amico',
                          'dwi.nii.gz')
    in_bval = os.path.join(get_home(), 'commit_amico',
                           'dwi.bval')
    in_bvec = os.path.join(get_home(), 'commit_amico',
                           'dwi.bvec')
    mask = os.path.join(get_home(), 'commit_amico',
                           'mask.nii.gz')
    fa = os.path.join(get_home(), 'commit_amico',
                          'fa.nii.gz')

    ret = script_runner.run('scil_compute_divide.py', '--in_dwi_linear',
                            in_dwi, '--in_bval_linear', in_bval,
                            '--in_bvec_linear', in_bvec,
                            '--mask', mask, '--fa', fa, '--do_weight_bvals',
                            '--do_weight_pa', '--do_multiple_s0',
                            '--redo_weight_bvals', '--processes', '1', '-f')
    assert ret.success