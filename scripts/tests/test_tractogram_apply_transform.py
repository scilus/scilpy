#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bst.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_apply_transform.py', '--help')
    assert ret.success


def test_execution_bst(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_model = os.path.join(get_home(), 'bst', 'template',
                            'rpt_m.trk')
    in_fa = os.path.join(get_home(), 'bst',
                         'fa.nii.gz')
    in_aff = os.path.join(get_home(), 'bst',
                          'output0GenericAffine.mat')
    in_warp = os.path.join(get_home(), 'bst',
                           'output1InverseWarp.nii.gz')
    ret = script_runner.run('scil_tractogram_apply_transform.py',
                            in_model, in_fa, in_aff, 'rpt_m_warp.trk',
                            '--inverse', '--in_deformation', in_warp,
                            '--cut')
    assert ret.success
