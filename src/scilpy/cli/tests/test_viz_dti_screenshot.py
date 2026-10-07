#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np
from PIL import Image

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_viz_dti_screenshot', '--help'])
    assert ret.success


def test_execution_viz_dti_screenshot(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_dwi = os.path.join(SCILPY_HOME, 'processing', 'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing', '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing', '1000.bvec')
    in_template = os.path.join(SCILPY_HOME, 'processing',
                               'mni_masked_2x2x2.nii.gz')
    out_dir = os.path.join(tmp_dir.name, 'out_exec')

    ret = script_runner.run(['scil_viz_dti_screenshot', in_dwi, in_bval,
                             in_bvec, in_template, '--out_dir', out_dir])
    assert ret.success
    assert os.path.exists(os.path.join(out_dir, 'axial.png'))


def test_non_ras_viz_dti_screenshot(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_dwi = os.path.join(SCILPY_HOME, 'processing', 'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing', '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing', '1000.bvec')
    in_template = os.path.join(SCILPY_HOME, 'processing',
                               'mni_masked_2x2x2.nii.gz')

    # Save LAS copy of template
    tpl_img = nib.load(in_template)
    tpl_data = tpl_img.get_fdata(dtype=np.float32)
    tpl_aff_las = tpl_img.affine.copy()
    tpl_aff_las[0, 0] *= -1
    tpl_aff_las[0, 3] = (tpl_img.affine[0, 3]
                         + (tpl_data.shape[0] - 1) * tpl_img.affine[0, 0])
    nib.save(nib.Nifti1Image(tpl_data[::-1].copy(), tpl_aff_las),
             'template_las.nii.gz')

    # Save LAS copy of DWI
    dwi_img = nib.load(in_dwi)
    dwi_data = dwi_img.get_fdata(dtype=np.float32)
    dwi_aff_las = dwi_img.affine.copy()
    dwi_aff_las[0, 0] *= -1
    dwi_aff_las[0, 3] = (dwi_img.affine[0, 3]
                         + (dwi_data.shape[0] - 1) * dwi_img.affine[0, 0])
    nib.save(nib.Nifti1Image(dwi_data[::-1].copy(), dwi_aff_las),
             'dwi_las.nii.gz')

    out_ras_dir = os.path.join(tmp_dir.name, 'out_ras')
    ret_ras = script_runner.run(['scil_viz_dti_screenshot', in_dwi, in_bval,
                                 in_bvec, in_template,
                                 '--out_dir', out_ras_dir])
    assert ret_ras.success

    out_las_dir = os.path.join(tmp_dir.name, 'out_las')
    ret_las = script_runner.run(['scil_viz_dti_screenshot', 'dwi_las.nii.gz',
                                 in_bval, in_bvec, 'template_las.nii.gz',
                                 '--out_dir', out_las_dir])
    assert ret_las.success

    for axis in ['axial', 'coronal', 'sagittal']:
        out_ras = os.path.join(out_ras_dir, f'{axis}.png')
        out_las = os.path.join(out_las_dir, f'{axis}.png')
        assert os.path.exists(out_las)

        rendered_ras = np.asarray(Image.open(out_ras)).astype(np.float32)
        rendered_las = np.asarray(Image.open(out_las)).astype(np.float32)

        # Assert non-trivial pixel variance to confirm glyphs are drawn at all.
        assert np.std(rendered_las) > 3.0

        # RAS and LAS describe the exact same anatomy on different on-disk
        # grids. If orientation/rotation is correctly applied before rendering,
        # both renders should be visually near-identical. Before the fix, the
        # LAS input ignored orientation/scale and would not match.
        diff = np.abs(rendered_ras - rendered_las)
        assert np.allclose(rendered_ras, rendered_las, atol=5.0), (
            f"LAS and RAS renders of {axis} differ too much (mean abs "
            f"diff = {np.mean(diff):.2f}); "
            f"orientation may not be applied correctly.")
