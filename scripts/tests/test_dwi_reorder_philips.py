#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import tempfile

from dipy.io.gradients import read_bvals_bvecs
import numpy as np

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_dwi_reorder_philips.py', '--help')
    assert ret.success


def test_reorder(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'processing', 'dwi_crop.nii.gz')
    in_bval = os.path.join(get_home(), 'processing', 'dwi.bval')
    in_bvec = os.path.join(get_home(), 'processing', 'dwi.bvec')
    table = np.ones((64, 4))
    bval, bvec = read_bvals_bvecs(in_bval, in_bvec)
    table[:, :3] = bvec
    table[:, 3] = bval
    tmp = np.copy(table[15])
    table[15] = table[30]
    table[30] = tmp
    in_table = os.path.expanduser(tmp_dir.name) + "/in_table.txt"
    np.savetxt(in_table, table, header="Test")
    ret = script_runner.run('scil_dwi_reorder_philips.py', in_dwi, in_bval,
                            in_bvec, in_table, 'out1', '-f')
    assert ret.success


def test_reorder_w_json_old_version(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'processing', 'dwi_crop.nii.gz')
    in_bval = os.path.join(get_home(), 'processing', 'dwi.bval')
    in_bvec = os.path.join(get_home(), 'processing', 'dwi.bvec')
    table = np.ones((64, 4))
    bval, bvec = read_bvals_bvecs(in_bval, in_bvec)
    table[:, :3] = bvec
    table[:, 3] = bval
    tmp = np.copy(table[15])
    table[15] = table[30]
    table[30] = tmp
    in_table = os.path.expanduser(tmp_dir.name) + "/in_table.txt"
    np.savetxt(in_table, table, header="Test")
    in_json = os.path.expanduser(tmp_dir.name) + "/in_json.json"
    info = {'SoftwareVersions': '5.5'}
    with open(in_json, 'w') as f:
        json.dump(info, f)
    ret = script_runner.run('scil_dwi_reorder_philips.py', in_dwi, in_bval,
                            in_bvec, in_table, 'out2', '--json', in_json)
    assert ret.success


def test_reorder_w_json_new_version(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'processing', 'dwi_crop.nii.gz')
    in_bval = os.path.join(get_home(), 'processing', 'dwi.bval')
    in_bvec = os.path.join(get_home(), 'processing', 'dwi.bvec')
    table = np.ones((64, 4))
    bval, bvec = read_bvals_bvecs(in_bval, in_bvec)
    table[:, :3] = bvec
    table[:, 3] = bval
    tmp = np.copy(table[15])
    table[15] = table[30]
    table[30] = tmp
    in_table = os.path.expanduser(tmp_dir.name) + "/in_table.txt"
    np.savetxt(in_table, table, header="Test")
    in_json = os.path.expanduser(tmp_dir.name) + "/in_json.json"
    info = {'SoftwareVersions': '5.6'}
    with open(in_json, 'w') as f:
        json.dump(info, f)
    ret = script_runner.run('scil_dwi_reorder_philips.py', in_dwi, in_bval,
                            in_bvec, in_table, 'out3', '--json', in_json)
    assert not ret.success
