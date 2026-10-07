#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import tempfile

import nibabel as nib
import numpy as np

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

    las_dir = os.path.join(tmp_dir.name, 'las')
    ras_dir = os.path.join(tmp_dir.name, 'ras')
    os.makedirs(las_dir, exist_ok=True)
    os.makedirs(ras_dir, exist_ok=True)

    suffixes = ['_signal_fraction', '_evals', '_isotropic',
                '_num_tensors', '_evecs']

    # Copy original LAS files and seed a known world-space unit vector
    known_vec = np.array([0.26726124, 0.53452248, 0.80178373],
                         dtype=np.float32)
    target_vox_las = (2, 4, 5)

    for i in range(1, 4):
        for suffix in suffixes:
            fn = f'sub-01_D{i}{suffix}.nii.gz'
            src = os.path.join(SCILPY_HOME, 'mrds', fn)
            dst = os.path.join(las_dir, fn)
            if i == 2 and suffix == '_evecs':
                img = nib.load(src)
                data = img.get_fdata(dtype=np.float32)
                data[target_vox_las[0], target_vox_las[1],
                     target_vox_las[2], 0:3] = known_vec
                nib.save(nib.Nifti1Image(data, img.affine, img.header), dst)
            else:
                shutil.copyfile(src, dst)

    shutil.copyfile(os.path.join(SCILPY_HOME, 'mrds', 'sub-01_nufo.nii.gz'),
                    os.path.join(las_dir, 'sub-01_nufo.nii.gz'))

    # Convert the LAS dataset with the seeded vector into genuine RAS fixtures
    for i in range(1, 4):
        for suffix in suffixes:
            fn_las = os.path.join(las_dir, f'sub-01_D{i}{suffix}.nii.gz')
            fn_ras = os.path.join(ras_dir, f'ras_D{i}{suffix}.nii.gz')
            simg = StatefulImage.load(fn_las)
            simg.to_ras()
            nib.save(nib.Nifti1Image(simg.get_fdata(), simg.affine), fn_ras)

    nufo_simg = StatefulImage.load(os.path.join(las_dir, 'sub-01_nufo.nii.gz'))
    nufo_simg.to_ras()
    nib.save(
        nib.Nifti1Image(nufo_simg.get_fdata().astype(np.uint8),
                        nufo_simg.affine),
        os.path.join(ras_dir, 'ras_nufo.nii.gz'))

    # Run on LAS files
    ret_las = script_runner.run([
        'scil_mrds_select_number_of_tensors',
        os.path.join(las_dir, 'sub-01'),
        os.path.join(las_dir, 'sub-01_nufo.nii.gz'),
        '--out_prefix', 'out_las', '-f'])
    assert ret_las.success

    # Run on RAS files
    ret_ras = script_runner.run([
        'scil_mrds_select_number_of_tensors',
        os.path.join(ras_dir, 'ras'),
        os.path.join(ras_dir, 'ras_nufo.nii.gz'),
        '--out_prefix', 'out_ras', '-f'])
    assert ret_ras.success

    # Verify LAS output orientation is preserved
    out_las_evecs = nib.load('out_las_MRDS_evecs.nii.gz')
    ax_las = nib.orientations.aff2axcodes(out_las_evecs.affine)
    assert ax_las == ('L', 'A', 'S')

    # Verify RAS output orientation is preserved
    out_ras_evecs = nib.load('out_ras_MRDS_evecs.nii.gz')
    ax_ras = nib.orientations.aff2axcodes(out_ras_evecs.affine)
    assert ax_ras == ('R', 'A', 'S')

    # Check known world-space vector is preserved in LAS output at seeded voxel
    out_las_data = out_las_evecs.get_fdata(dtype=np.float32)
    assert np.allclose(
        out_las_data[target_vox_las[0], target_vox_las[1],
                     target_vox_las[2], 0:3],
        known_vec, atol=1e-5)

    # Check corresponding voxel in RAS output points in same world direction
    p_world = nib.affines.apply_affine(out_las_evecs.affine, target_vox_las)
    target_vox_ras = tuple(np.round(
        nib.affines.apply_affine(np.linalg.inv(out_ras_evecs.affine), p_world)
    ).astype(int))
    out_ras_data = out_ras_evecs.get_fdata(dtype=np.float32)
    assert np.allclose(
        out_ras_data[target_vox_ras[0], target_vox_ras[1],
                     target_vox_ras[2], 0:3],
        known_vec, atol=1e-5)

    # Compare all outputs across representations when reoriented to RAS
    for suffix in suffixes:
        las_simg = StatefulImage.load(f'out_las_MRDS{suffix}.nii.gz')
        las_simg.to_ras()
        ras_simg = StatefulImage.load(f'out_ras_MRDS{suffix}.nii.gz')
        assert np.allclose(ras_simg.get_fdata(), las_simg.get_fdata(),
                           atol=1e-5)
