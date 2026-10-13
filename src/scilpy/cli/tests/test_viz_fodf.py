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
    ret = script_runner.run(['scil_viz_fodf', '--help'])
    assert ret.success


def test_silent_without_output(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz')

    ret = script_runner.run(['scil_viz_fodf', in_fodf, '--silent'])

    # Should say that requires an output with --silent mode
    assert (not ret.success)


def test_run(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'processing', 'seed.nii.gz')
    # No variance file in our test data, but faking it with the fodf file.
    in_variance = os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz')
    out_name = os.path.join(tmp_dir.name, 'out.png')
    ret = script_runner.run(['scil_viz_fodf', in_fodf, '--silent',
                             '--in_transparency_mask', in_mask,
                             '--mask', in_mask,
                             '--variance', in_variance,
                             '--output', out_name])
    assert ret.success


def test_run_sphsubdivide(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'processing', 'seed.nii.gz')
    out_name = os.path.join(tmp_dir.name, 'out2.png')

    # Note. Cannot add --sph_subdivide to the test above, causes a memory
    # crash. Without the variance, lighter.
    ret = script_runner.run(['scil_viz_fodf', in_fodf, '--silent',
                             '--mask', in_mask,
                             '--sph_subdivide', '2',
                             '--sphere', 'repulsion100',
                             '--output', out_name])

    assert ret.success


def test_non_ras_viz_fodf(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_dummy = os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz')
    img = nib.load(in_dummy)
    data = img.get_fdata(dtype=np.float32)

    # Save a RAS copy of the exact same anatomy, used below as the ground
    # truth to compare the LAS render against.
    aff_ras = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data, aff_ras), 'fodf_ras.nii.gz')

    # Save LAS copy
    aff_las = np.diag([-2.0, 2.0, 2.0, 1.0])
    aff_las[0, 3] = 20.0
    data_las = data[::-1].copy()
    nib.save(nib.Nifti1Image(data_las, aff_las), 'fodf_las.nii.gz')

    out_ras = os.path.join(tmp_dir.name, 'test_fodf_ras.png')
    ret_ras = script_runner.run(['scil_viz_fodf', 'fodf_ras.nii.gz',
                                 '--silent', '--output', out_ras])
    assert ret_ras.success

    out_las = os.path.join(tmp_dir.name, 'test_fodf_las.png')
    ret_las = script_runner.run(['scil_viz_fodf', 'fodf_las.nii.gz',
                                 '--silent', '--output', out_las])
    assert ret_las.success
    assert os.path.exists(out_las)

    rendered_ras = np.asarray(Image.open(out_ras)).astype(np.float32)
    rendered_las = np.asarray(Image.open(out_las)).astype(np.float32)

    # Assert non-trivial pixel variance to confirm glyphs are drawn at all.
    assert np.std(rendered_las) > 3.0

    # RAS and LAS describe the exact same anatomy on different on-disk
    # grids. If orientation/rotation is correctly applied before rendering,
    # both renders should be visually near-identical.
    diff = np.abs(rendered_ras - rendered_las)
    assert np.allclose(rendered_ras, rendered_las, atol=5.0), (
        f"LAS and RAS renders of the same data differ too much (mean abs "
        f"diff = {np.mean(diff):.2f}); "
        f"orientation may not be applied correctly.")
