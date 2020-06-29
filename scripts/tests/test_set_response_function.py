#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_set_response_function.py', '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_frf = os.path.join(get_home(), 'processing',
                          'frf.txt')
    ret = script_runner.run('scil_set_response_function.py', in_frf, '15,4,4',
                            'nfrf.txt')
    assert ret.success
