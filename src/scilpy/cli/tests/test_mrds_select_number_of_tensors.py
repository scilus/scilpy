#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np
from PIL import Image

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.io.stateful_image import StatefulImage

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['mrds.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_mrds_select_number_of_tensors', '--help'])
    assert ret.success


def test_execution_mrds(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_nufo = os.path.join(SCILPY_HOME,
                           'mrds', 'sub-01_nufo.nii.gz')
    # no option
    ret = script_runner.run(['scil_mrds_select_number_of_tensors',
                             SCILPY_HOME + '/mrds/sub-01',
                             in_nufo,
                             '-f'])
    assert ret.success


def test_execution_mrds_w_mask(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_nufo = os.path.join(SCILPY_HOME,
                           'mrds', 'sub-01_nufo.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'mrds',
                           'sub-01_mask.nii.gz')

    ret = script_runner.run(['scil_mrds_select_number_of_tensors',
                             SCILPY_HOME + '/mrds/sub-01',
                             in_nufo,
                             '--mask', in_mask,
                             '-f'])
    assert ret.success


def test_non_ras_mrds_select_number_of_tensors(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    nufo_path = os.path.join(SCILPY_HOME, 'mrds', 'sub-01_nufo.nii.gz')
    nufo_img = nib.load(nufo_path)
    prefix = os.path.join(SCILPY_HOME, 'mrds', 'sub-01')

    # sub-01 MRDS data is in LAS orientation.
    # Create an axial mid-slice mask in LAS.
    mid_z = nufo_img.shape[2] // 2
    mask_las = np.zeros(nufo_img.shape, dtype=np.uint8)
    mask_las[:, :, mid_z] = 1
    las_mask_path = 'mask_midslice_las.nii.gz'
    nib.save(nib.Nifti1Image(mask_las, nufo_img.affine), las_mask_path)

    # Reorient mask to coronal/sagittal axis order (PIR)
    simg_mask = StatefulImage.convert_to_simg(
        nib.Nifti1Image(mask_las, nufo_img.affine))
    simg_mask.reorient(('P', 'I', 'R'))
    pir_mask_path = 'mask_midslice_pir.nii.gz'
    nib.save(nib.Nifti1Image(simg_mask.get_fdata().astype(np.uint8),
                             simg_mask.affine),
             pir_mask_path)

    # Reference run with native LAS mask
    ret_ref = script_runner.run([
        'scil_mrds_select_number_of_tensors', prefix, nufo_path,
        '--mask', las_mask_path,
        '--out_prefix', 'out_ref', '-f'])
    assert ret_ref.success

    # Test run with cross-orientation PIR mask
    ret_test = script_runner.run([
        'scil_mrds_select_number_of_tensors', prefix, nufo_path,
        '--mask', pir_mask_path,
        '--out_prefix', 'out_test', '-f'])
    assert ret_test.success

    # Verify LAS output orientation is preserved
    out_num_img = nib.load('out_test_MRDS_num_tensors.nii.gz')
    assert nib.orientations.aff2axcodes(out_num_img.affine) == ('L', 'A', 'S')
    out_num = out_num_img.get_fdata()

    # In LAS, the mid-slice along z should be kept (non-zero),
    # while other slices should be masked out (zero).
    assert np.sum(out_num[:, :, mid_z]) > 0
    assert np.all(out_num[:, :, mid_z - 1] == 0)

    # Verify float output is not quantized to uint8 despite
    # uint8 NUFO reference
    out_sf = nib.load('out_test_MRDS_signal_fraction.nii.gz')
    assert out_sf.header.get_data_dtype() == np.float32
    sf_data = out_sf.get_fdata()
    assert np.any((sf_data > 0) & (sf_data < 1))

    # Verify rendered peaks screenshots match reference
    ret_viz_ref = script_runner.run([
        'scil_viz_volume_screenshot', nufo_path, 'ref_peaks.png',
        '--peaks', 'out_ref_MRDS_evecs.nii.gz',
        '--slices', str(mid_z),
        '--volume_opacity', '0.0', '-f'])
    assert ret_viz_ref.success

    ret_viz_test = script_runner.run([
        'scil_viz_volume_screenshot', nufo_path, 'test_peaks.png',
        '--peaks', 'out_test_MRDS_evecs.nii.gz',
        '--slices', str(mid_z),
        '--volume_opacity', '0.0', '-f'])
    assert ret_viz_test.success

    ref_arr = np.asarray(
        Image.open(f'ref_peaks_slice_{mid_z}.png')).astype(np.float32)
    test_arr = np.asarray(
        Image.open(f'test_peaks_slice_{mid_z}.png')).astype(np.float32)

    # Confirm peak glyphs are rendered with non-trivial pixel variance
    assert np.std(test_arr) > 3.0

    # Confirm visual equivalence between reference and reoriented mask
    diff = np.abs(ref_arr - test_arr)
    assert np.allclose(ref_arr, test_arr, atol=1.0), (
        f"Peaks render with reoriented mask differs from reference "
        f"(max diff = {np.max(diff):.2f})")
