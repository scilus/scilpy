#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['btensor_testdata.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_fodf_memsmt.py', '--help')
    assert ret.success


def test_inputs_check(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'dwi_planar.nii.gz')
    in_bval_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvals')
    in_bvec_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvecs')
    in_wm_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                             'wm_frf.txt')
    in_gm_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                             'gm_frf.txt')
    in_csf_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'csf_frf.txt')

    ret = script_runner.run('scil_fodf_memsmt.py', in_wm_frf,
                            in_gm_frf, in_csf_frf, '--in_dwis',
                            in_dwi_lin, in_dwi_plan, '--in_bvals',
                            in_bval_lin, '--in_bvecs', in_bvec_lin,
                            '--in_bdeltas', '1',
                            '--wm_out_fODF', 'wm_fodf.nii.gz',
                            '--gm_out_fODF', 'gm_fodf.nii.gz',
                            '--csf_out_fODF', 'csf_fodf.nii.gz', '--vf',
                            'vf.nii.gz', '--sh_order', '4', '--sh_basis',
                            'tournier07', '--processes', '1', '-f')
    assert (not ret.success)

    ret = script_runner.run('scil_fodf_memsmt.py', in_wm_frf,
                            in_gm_frf, in_csf_frf, '--in_dwis',
                            in_dwi_lin, in_dwi_plan, '--in_bvals',
                            in_bval_lin, in_bval_plan, '--in_bvecs',
                            in_bvec_lin, in_bvec_plan, '--in_bdeltas',
                            '1', '-0.5', '0',
                            '--wm_out_fODF', 'wm_fodf.nii.gz',
                            '--gm_out_fODF', 'gm_fodf.nii.gz',
                            '--csf_out_fODF', 'csf_fodf.nii.gz', '--vf',
                            'vf.nii.gz', '--sh_order', '4', '--sh_basis',
                            'tournier07', '--processes', '1', '-f')
    assert (not ret.success)


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_spherical.nii.gz')
    in_bval_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvals')
    in_bvec_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvecs')
    in_wm_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                             'wm_frf.txt')
    in_gm_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                             'gm_frf.txt')
    in_csf_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'csf_frf.txt')

    ret = script_runner.run('scil_fodf_memsmt.py', in_wm_frf,
                            in_gm_frf, in_csf_frf, '--in_dwis',
                            in_dwi_lin, in_dwi_sph, '--in_bvals',
                            in_bval_lin, in_bval_sph,
                            '--in_bvecs', in_bvec_lin,
                            in_bvec_sph, '--in_bdeltas', '1', '0',
                            '--sh_order', '8', '--processes', '8', '-f')
    assert ret.success
