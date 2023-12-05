#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_local_tracking_dev.py', '--help')
    assert ret.success
