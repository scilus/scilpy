#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['btensor_testdata.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_btensor_metrics.py', '--help')
    assert ret.success


def test_nb_btensors_check(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(get_home(), 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(get_home(), 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(get_home(), 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_plan = os.path.join(get_home(), 'btensor_testdata',
                               'dwi_planar.nii.gz')
    in_bval_plan = os.path.join(get_home(), 'btensor_testdata',
                                'planar.bvals')
    in_bvec_plan = os.path.join(get_home(), 'btensor_testdata',
                                'planar.bvecs')
    fa = os.path.join(get_home(), 'btensor',
                      'fa.nii.gz')

    ret = script_runner.run('scil_btensor_metrics.py', '--in_dwis',
                            in_dwi_lin, '--in_bvals', in_bval_lin,
                            '--in_bvecs', in_bvec_lin, '--in_bdeltas', '1',
                            '--fa', fa, '--do_weight_bvals',
                            '--do_weight_pa', '--do_multiple_s0',
                            '--processes', '1', '-f')
    assert (not ret.success)

    ret = script_runner.run('scil_btensor_metrics.py', '--in_dwis',
                            in_dwi_lin, in_dwi_plan, '--in_bvals', in_bval_lin,
                            in_bval_plan, '--in_bvecs', in_bvec_lin,
                            in_bvec_plan, '--in_bdeltas', '1', '1',
                            '--fa', fa, '--do_weight_bvals',
                            '--do_weight_pa', '--do_multiple_s0',
                            '--processes', '1', '-f')
    assert (not ret.success)


def test_inputs_check(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(get_home(), 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(get_home(), 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(get_home(), 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_plan = os.path.join(get_home(), 'btensor_testdata',
                               'dwi_planar.nii.gz')
    in_bval_plan = os.path.join(get_home(), 'btensor_testdata',
                                'planar.bvals')
    in_bvec_plan = os.path.join(get_home(), 'btensor_testdata',
                                'planar.bvecs')
    fa = os.path.join(get_home(), 'btensor',
                      'fa.nii.gz')

    ret = script_runner.run('scil_btensor_metrics.py', '--in_dwis',
                            in_dwi_lin, in_dwi_plan, '--in_bvals', in_bval_lin,
                            '--in_bvecs', in_bvec_lin, '--in_bdeltas', '1',
                            '--fa', fa, '--do_weight_bvals',
                            '--do_weight_pa', '--do_multiple_s0',
                            '--processes', '1', '-f')
    assert (not ret.success)

    ret = script_runner.run('scil_btensor_metrics.py', '--in_dwis',
                            in_dwi_lin, in_dwi_plan, '--in_bvals',
                            in_bval_lin, in_bval_plan, '--in_bvecs',
                            in_bvec_lin, in_bvec_plan, '--in_bdeltas', '1',
                            '-0.5', '0', '--fa', fa, '--do_weight_bvals',
                            '--do_weight_pa', '--do_multiple_s0',
                            '--processes', '1', '-f')
    assert (not ret.success)


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(get_home(), 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(get_home(), 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(get_home(), 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_plan = os.path.join(get_home(), 'btensor_testdata',
                               'dwi_planar.nii.gz')
    in_bval_plan = os.path.join(get_home(), 'btensor_testdata',
                                'planar.bvals')
    in_bvec_plan = os.path.join(get_home(), 'btensor_testdata',
                                'planar.bvecs')
    in_dwi_sph = os.path.join(get_home(), 'btensor_testdata',
                              'dwi_spherical.nii.gz')
    in_bval_sph = os.path.join(get_home(), 'btensor_testdata',
                               'spherical.bvals')
    in_bvec_sph = os.path.join(get_home(), 'btensor_testdata',
                               'spherical.bvecs')
    fa = os.path.join(get_home(), 'btensor_testdata',
                      'fa.nii.gz')

    ret = script_runner.run('scil_btensor_metrics.py', '--in_dwis',
                            in_dwi_lin, in_dwi_plan, in_dwi_sph,
                            '--in_bvals', in_bval_lin, in_bval_plan,
                            in_bval_sph, '--in_bvecs', in_bvec_lin,
                            in_bvec_plan, in_bvec_sph, '--in_bdeltas',
                            '1', '-0.5', '0', '--fa', fa, '--do_weight_bvals',
                            '--do_weight_pa', '--do_multiple_s0',
                            '--processes', '1', '-f')
    assert (ret.success)
