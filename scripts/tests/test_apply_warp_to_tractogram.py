#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bst.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_apply_warp_to_tractogram.py', '--help')
    assert ret.success


def test_execution_bst(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_bundle = os.path.join(get_home(), 'bst',
                                'rpt_m_lin.trk')
    input_fa = os.path.join(get_home(), 'bst',
                            'fa.nii.gz')
    input_warp = os.path.join(get_home(), 'bst',
                              'output1InverseWarp.nii.gz')
    ret = script_runner.run('scil_apply_warp_to_tractogram.py', input_bundle,
                            input_fa, input_warp,
                            'rpt_m_warp.trk', '--cut')
    assert ret.success
