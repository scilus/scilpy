#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import nibabel as nib
import numpy as np
import os
import pytest
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict


def test_help_option(script_runner):
    ret = script_runner.run('scil_lesions_info.py', '--help')
    assert ret.success