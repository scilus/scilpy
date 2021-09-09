#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import numpy as np

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_memsmt_fodf.py', '--help')
    assert ret.success


def test_execution_processing(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(get_home(), 'commit_amico',
                          'dwi.nii.gz')
    in_bval = os.path.join(get_home(), 'commit_amico',
                           'dwi.bval')
    in_bvec = os.path.join(get_home(), 'commit_amico',
                           'dwi.bvec')
    in_wm_frf = os.path.join(get_home(), 'commit_amico',
                          'wm_frf.txt')
    in_gm_frf = os.path.join(get_home(), 'commit_amico',
                          'gm_frf.txt')
    in_csf_frf = os.path.join(get_home(), 'commit_amico',
                          'csf_frf.txt')
    out_wm_frf = os.path.join(get_home(), 'commit_amico',
                          'wm_frf_btens.txt')
    out_gm_frf = os.path.join(get_home(), 'commit_amico',
                          'gm_frf_btens.txt')
    out_csf_frf = os.path.join(get_home(), 'commit_amico',
                          'csf_frf_btens.txt')
    mask = os.path.join(get_home(), 'commit_amico',
                           'mask.nii.gz')
    wm_frf = np.loadtxt(in_wm_frf)
    wm_frf = wm_frf[0:3]
    np.savetxt(in_wm_frf, wm_frf)
    wm_frf_btensor = np.zeros((wm_frf.shape[0] * 3, wm_frf.shape[1]))
    wm_frf_btensor[0:3] = wm_frf
    wm_frf_btensor[3:6] = wm_frf
    wm_frf_btensor[6:9] = wm_frf
    np.savetxt(out_wm_frf, wm_frf_btensor)

    gm_frf = np.loadtxt(in_gm_frf)
    gm_frf = gm_frf[0:3]
    np.savetxt(in_gm_frf, gm_frf)
    gm_frf_btensor = np.zeros((gm_frf.shape[0] * 3, gm_frf.shape[1]))
    gm_frf_btensor[0:3] = gm_frf
    gm_frf_btensor[3:6] = gm_frf
    gm_frf_btensor[6:9] = gm_frf
    np.savetxt(out_gm_frf, gm_frf_btensor)

    csf_frf = np.loadtxt(in_csf_frf)
    csf_frf = csf_frf[0:3]
    np.savetxt(in_csf_frf, csf_frf)
    csf_frf_btensor = np.zeros((csf_frf.shape[0] * 3, csf_frf.shape[1]))
    csf_frf_btensor[0:3] = csf_frf
    csf_frf_btensor[3:6] = csf_frf
    csf_frf_btensor[6:9] = csf_frf
    np.savetxt(out_csf_frf, csf_frf_btensor)

    ret = script_runner.run('scil_compute_memsmt_fodf.py', out_wm_frf,
                            out_gm_frf, out_csf_frf, '--in_dwi_linear',
                            in_dwi, '--in_bval_linear', in_bval,
                            '--in_bvec_linear', in_bvec, '--in_dwi_planar',
                            in_dwi, '--in_bval_planar', in_bval,
                            '--in_bvec_planar', in_bvec, '--in_dwi_spherical',
                            in_dwi, '--in_bval_spherical', in_bval,
                            '--in_bvec_spherical', in_bvec,
                            '--mask', mask, '--wm_out_fODF', 'wm_fodf.nii.gz',
                            '--gm_out_fODF', 'gm_fodf.nii.gz', '--csf_out_fODF',
                            'csf_fodf.nii.gz', '--vf', 'vf.nii.gz',
                            '--sh_order', '4', '--sh_basis', 'tournier07',
                            '--processes', '1', '-f')
    assert ret.success