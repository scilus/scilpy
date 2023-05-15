#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['stats.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_pca.py',
                            '--help')
    assert ret.success


def test_execution_pca(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_folder = os.path.join(get_home(), 'stats/pca')
    output_folder = os.path.join(get_home(), 'stats/pca_out')
    ids = os.path.join(get_home(), 'stats/pca',
                       'list_id.txt')
    ret = script_runner.run('scil_compute_pca.py', input_folder, output_folder, '--metrics', 'ad',
                            'fa', 'md', 'rd', 'nufo', 'afd_total', 'afd_fixel', '--list_ids',
                            ids, '-f')
    assert ret.success
