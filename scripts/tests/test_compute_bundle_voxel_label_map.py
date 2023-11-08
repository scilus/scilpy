#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_bundle_voxel_label_map.py', '--help')
    assert ret.success


def test_execution_tractometry_euclidian(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'tractometry',
                             'IFGWM.trk')
    in_centroid = os.path.join(get_home(), 'tractometry',
                               'IFGWM_uni_c_10.trk')
    ret = script_runner.run('scil_compute_bundle_voxel_label_map.py',
                            in_bundle, in_centroid,
                            'results_euc/',
                            '--colormap', 'viridis')
    assert ret.success

def test_execution_tractometry_hyperplane(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'tractometry',
                             'IFGWM.trk')
    in_centroid = os.path.join(get_home(), 'tractometry',
                               'IFGWM_uni_c_10.trk')
    ret = script_runner.run('scil_compute_bundle_voxel_label_map.py',
                            in_bundle, in_centroid,
                            'results_man/',
                            '--colormap', 'viridis',
                            '--hyperplane', '--use_manhattan')
    assert ret.success
