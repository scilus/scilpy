#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bst.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_todi.py', '--help')
    assert ret.success


def test_execution_bst(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'bst',
                             'rpt_m_warp.trk')
    in_mask = os.path.join(get_home(), 'bst',
                           'mask.nii.gz')
    ret = script_runner.run('scil_compute_todi.py', in_bundle, '--mask',
                            in_mask, '--out_mask', 'todi_mask.nii.gz',
                            '--out_tdi', 'tdi.nii.gz',
                            '--out_todi_sh', 'todi_sh.nii.gz',
                            '--sh_order', '6',
                            '--normalize_per_voxel', '--smooth_todi',
                            '--sh_basis', 'descoteaux07')
    assert ret.success


def test_execution_asym(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_bundle = os.path.join(get_home(), 'bst',
                             'rpt_m_warp.trk')
    ret = script_runner.run('scil_compute_todi.py', in_bundle,
                            '--out_todi_sh', 'atodi_sh_8.nii.gz',
                            '--asymmetric', '--n_steps', '2')

    assert ret.success
