#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_execute_asymmetric_filtering.py',
                            '--help')
    assert ret.success


def test_asym_basis_output(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'processing',
                           'fodf_descoteaux07_sub.nii.gz')

    # We use a low resolution sphere to reduce execution time
    ret = script_runner.run('scil_execute_asymmetric_filtering.py', in_fodf,
                            'out_0.nii.gz', '--sphere', 'repulsion100')
    assert ret.success


def test_sym_basis_output(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'processing',
                           'fodf_descoteaux07_sub.nii.gz')

    # We use a low resolution sphere to reduce execution time
    ret = script_runner.run('scil_execute_asymmetric_filtering.py', in_fodf,
                            'out_1.nii.gz', '--out_sym', 'out_sym.nii.gz',
                            '--sphere', 'repulsion100')
    assert ret.success


def test_asym_input(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'processing',
                           'fodf_descoteaux07_sub_full.nii.gz')

    # We use a low resolution sphere to reduce execution time
    ret = script_runner.run('scil_execute_asymmetric_filtering.py', in_fodf,
                            'out_2.nii.gz', '--sphere', 'repulsion100', '-f')
    assert ret.success
