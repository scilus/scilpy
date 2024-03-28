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
    ret = script_runner.run('scil_sh_to_sf.py', '--help')
    assert ret.success


def test_execution_in_sphere(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sh = os.path.join(SCILPY_HOME, 'processing', 'sh_1000.nii.gz')
    in_b0 = os.path.join(SCILPY_HOME, 'processing', 'fa.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing', '1000.bval')

    # Required: either --sphere or --in_bvec. Here, --sphere
    ret = script_runner.run('scil_sh_to_sf.py', in_sh,
                            'sf_724.nii.gz', '--in_bval',
                            in_bval, '--in_b0', in_b0, '--out_bval',
                            'sf_724.bval', '--out_bvec', 'sf_724.bvec',
                            '--sphere', 'symmetric724', '--dtype', 'float32')
    assert ret.success


def test_execution_in_bvec(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sh = os.path.join(SCILPY_HOME, 'processing', 'sh_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing', '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing', '1000.bvec')

    # --in_bvec: in_bval is required.
    ret = script_runner.run('scil_sh_to_sf.py', in_sh,
                            'sf_724.nii.gz', '--in_bval', in_bval,
                            '--out_bval', 'sf_724.bval',
                            '--out_bvec', 'sf_724.bvec',
                            '--in_bvec', in_bvec, '--dtype', 'float32', '-f')
    assert ret.success

    # Test that fails if no bvals is given.
    ret = script_runner.run('scil_sh_to_sf.py', in_sh,
                            'sf_724.nii.gz',
                            '--out_bvec', 'sf_724.bvec',
                            '--in_bvec', in_bvec, '--dtype', 'float32', '-f')
    assert not ret.success


def test_execution_no_bval(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sh = os.path.join(SCILPY_HOME, 'processing', 'sh_1000.nii.gz')
    in_b0 = os.path.join(SCILPY_HOME, 'processing', 'fa.nii.gz')

    # --sphere but no --bval
    ret = script_runner.run('scil_sh_to_sf.py', in_sh,
                            'sf_724.nii.gz', '--in_b0', in_b0,
                            '--out_bvec', 'sf_724.bvec', '--b0_scaling',
                            '--sphere', 'symmetric724', '--dtype', 'float32',
                            '-f')
    assert ret.success
