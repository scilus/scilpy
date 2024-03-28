#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_fodf_metrics.py', '--help')
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing',
                           'fodf_descoteaux07.nii.gz')
    ret = script_runner.run('scil_fodf_metrics.py', in_fodf, '--not_al',
                            '--peaks', 'peaks.nii.gz',
                            '--afd_max', 'afd_max.nii.gz',
                            '--afd_total', 'afd_tot.nii.gz',
                            '--afd_sum', 'afd_sum.nii.gz',
                            '--nufo', 'nufo.nii.gz', '--processes', '1')
    assert ret.success
