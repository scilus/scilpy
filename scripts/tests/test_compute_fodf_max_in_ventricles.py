#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_fodf_max_in_ventricles.py', '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(get_home(), 'processing',
                           'fodf.nii.gz')
    in_fa = os.path.join(get_home(), 'processing',
                         'fa.nii.gz')
    in_md = os.path.join(get_home(), 'processing',
                         'md.nii.gz')
    ret = script_runner.run('scil_compute_fodf_max_in_ventricles.py', in_fodf,
                            in_fa, in_md, '--sh_basis', 'tournier07')
    assert ret.success
