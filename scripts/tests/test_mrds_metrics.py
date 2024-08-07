#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict


def test_help_option(script_runner):
    ret = script_runner.run('scil_mrds_metrics.py', '--help')
    assert ret.success
