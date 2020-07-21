#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['others.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_remove_similar_streamlines.py', '--help')
    assert ret.success


def test_execution_others(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'others',
                             'IFGWM.trk')
    ret = script_runner.run('scil_remove_similar_streamlines.py', in_bundle,
                            '2', 'IFGWM_sub.trk', '--min_cluster_size', '2',
                            '--clustering_thr', '8', '--avg_similar',
                            '--processes', '1')
    assert ret.success
