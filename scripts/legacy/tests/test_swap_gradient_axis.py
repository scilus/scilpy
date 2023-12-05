#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict


def test_help_option(script_runner):
    ret = script_runner.run('scil_swap_gradient_axis.py', '--help')
    assert ret.success
