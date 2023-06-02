#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bst.zip'])
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_apply_transform_to_bvecs.py', '--help')
    assert ret.success


def test_execution_bst(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bvecs = os.path.join(get_home(), 'processing',
                           'dwi.bvec')
    in_aff = os.path.join(get_home(), 'bst',
                          'output0GenericAffine.mat')
    ret = script_runner.run('scil_apply_transform_to_bvecs.py',
                            in_bvecs, in_aff,
                            'bvecs_transformed.bvec', '--inverse')
    assert ret.success
