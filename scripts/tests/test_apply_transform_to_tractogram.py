#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bst.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_apply_transform_to_tractogram.py', '--help')
    assert ret.success


def test_execution_bst(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_model = os.path.join(get_home(), 'bst', 'template',
                               'rpt_m.trk')
    input_fa = os.path.join(get_home(), 'bst',
                            'fa.nii.gz')
    input_aff = os.path.join(get_home(), 'bst',
                             'output0GenericAffine.mat')
    script_runner._save_and_reset_logger()
    ret = script_runner.run('scil_apply_transform_to_tractogram.py',
                            input_model, input_fa, input_aff, 'rpt_m_lin.trk',
                            '--inverse', '--cut')
    assert ret.success
