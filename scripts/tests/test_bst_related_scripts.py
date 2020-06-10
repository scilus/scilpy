#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict())
tmp_dir = tempfile.TemporaryDirectory()


def test_transform_trk(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_model = os.path.join(get_home(), 'bst', 'template',
                               'rpt_m.trk')
    input_fa = os.path.join(get_home(), 'bst',
                            'fa.nii.gz')
    input_aff = os.path.join(get_home(), 'bst',
                             'output0GenericAffine.mat')
    ret = script_runner.run('scil_apply_transform_to_tractogram.py',
                            input_model, input_fa, input_aff, 'rpt_m_lin.trk',
                            '--inverse', '--cut')
    return False


def test_transform_nii(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_model = os.path.join(get_home(), 'bst', 'template',
                               'template0.nii.gz')
    input_fa = os.path.join(get_home(), 'bst',
                            'fa.nii.gz')
    input_aff = os.path.join(get_home(), 'bst',
                             'output0GenericAffine.mat')
    ret = script_runner.run('scil_apply_transform_to_image.py',
                            input_model, input_fa, input_aff,
                            'template_lin.nii.gz', '--inverse')
    return ret.success


def test_warp_trk(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_fa = os.path.join(get_home(), 'bst',
                            'fa.nii.gz')
    input_warp = os.path.join(get_home(), 'bst',
                              'output1InverseWarp.nii.gz')
    ret = script_runner.run('scil_apply_warp_to_tractogram.py', 'rpt_m_lin.trk',
                            input_fa, 'output1InverseWarp.nii.gz',
                            'rpt_m_warp.trk', '--cut')
    return ret.success


def test_warp_trk(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_fa = os.path.join(get_home(), 'bst',
                            'fa.nii.gz')
    input_warp = os.path.join(get_home(), 'bst',
                              'output1InverseWarp.nii.gz')
    ret = script_runner.run('scil_apply_warp_to_tractogram.py', 'rpt_m_lin.trk',
                            input_fa, 'output1InverseWarp.nii.gz',
                            'rpt_m_warp.trk', '--cut')
    return ret.success


def test_todi(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_mask = os.path.join(get_home(), 'bst',
                              'mask.nii.gz')
    ret = script_runner.run('scil_compute_todi.py', 'rpt_m_warp.trk', '--mask',
                            'mask.nii.gz', '--out_mask', 'todi_mask.nii.gz',
                            '--out_lw_tdi', 'out_lw_tdi.nii.gz',
                            '--out_lw_todi_sh', 'lw_todi_sh.nii.gz',
                            '--sh_order' '6', '--sh_normed', '--smooth',
                            '--sh_basis', 'descoteaux07')
    return ret.success


def test_bst_priors(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_fodf = os.path.join(get_home(), 'bst',
                              'fodf.nii.gz')
    input_mask = os.path.join(get_home(), 'bst',
                              'mask.nii.gz')
    ret = script_runner.run('scil_generate_priors_from_bundle.py',
                            'rpt_m_lin.trk', input_fodf, input_mask,
                            '--todi_sigma', '1', '--output_dir', 'rpt_m/',
                            '--sh_basis', 'descoteaux07')
    return ret.success
