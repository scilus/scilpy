#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from scipy.ndimage import map_coordinates

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.io.utils import load_ants_warp, load_matrix_in_any_format

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bst.zip'])
tmp_dir = tempfile.TemporaryDirectory()

# In bst, the ANTs registration used fixed = fa, moving = template.
in_fa = os.path.join(SCILPY_HOME, 'bst', 'fa.nii.gz')
in_mask = os.path.join(SCILPY_HOME, 'bst', 'mask.nii.gz')
in_template = os.path.join(SCILPY_HOME, 'bst', 'template', 'template0.nii.gz')
in_aff = os.path.join(SCILPY_HOME, 'bst', 'output0GenericAffine.mat')
in_inv_warp = os.path.join(SCILPY_HOME, 'bst', 'output1InverseWarp.nii.gz')

# Two peaks per voxel (voxel frame of the moving image), with amplitudes.
PEAKS_VOX = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
AMPLITUDES = np.array([1.0, 0.5])


def _get_rotation(affine):
    m = affine[:3, :3]
    return m / np.linalg.norm(m, axis=0)


def _save_constant_peaks(ref_path, out_path):
    """Save the same two peaks in every voxel of the grid of ref_path."""
    ref = nib.load(ref_path)
    vecs = (PEAKS_VOX * AMPLITUDES[:, None]).ravel()
    data = np.broadcast_to(vecs, ref.shape[:3] + (6,)).astype(np.float32)
    nib.save(nib.Nifti1Image(data, ref.affine), out_path)


def _build_forward_warp(in_inv_warp, out_warp, n_iter=50, tol=1e-4):
    """
    Build the ANTs forward warp (Warp.nii.gz) from the inverse warp.

    ANTs maps fixed -> moving points as y = A(x + u(x)) and moving -> fixed
    points as x = z + v(z), with z = inv(A)(y). Both displacement fields u
    and v live on the fixed grid, in the space between the affine and the
    deformation, so the affine cancels out: u is the inverse of the
    displacement field v alone, i.e. x + u(x) = z with z + v(z) = x, solved
    by the fixed-point iteration u(x) = -v(x + u(x)).
    """
    img = nib.load(in_inv_warp)
    shape = img.shape[:3]
    v = img.get_fdata().reshape(shape + (3,))  # LPS displacements (mm)

    # Work in LPS, like the field itself.
    vox2lps = img.affine.copy()
    vox2lps[:2] *= -1
    lps2vox = np.linalg.inv(vox2lps)
    grid = np.indices(shape).transpose(1, 2, 3, 0).astype(np.float64)
    x = apply_affine(vox2lps, grid)

    def sample_v(pts):
        coords = apply_affine(lps2vox, pts).reshape(-1, 3).T
        return np.stack([map_coordinates(v[..., k], coords, order=1,
                                         mode='nearest')
                         for k in range(3)], axis=-1).reshape(pts.shape)

    u = -v
    for _ in range(n_iter):
        u_new = -sample_v(x + u)
        converged = np.abs(u_new - u).max() < tol
        u = u_new
        if converged:
            break

    # Sanity check: following u then v brings every point back to itself.
    residual = np.linalg.norm(u + sample_v(x + u), axis=-1)
    assert residual.max() < 1e-3, residual.max()

    nib.save(nib.Nifti1Image(u.reshape(img.shape), img.affine, img.header),
             out_warp)


def _expected_affine_peaks():
    """
    Peaks expected in the fa voxel frame for an affine-only transform: the
    fixed -> moving Jacobian is the linear part L of the RAS affine, so the
    moving world vectors are mapped back with inv(L).
    """
    lin = load_matrix_in_any_format(in_aff)[:3, :3]
    v_world = PEAKS_VOX @ _get_rotation(nib.load(in_template).affine).T
    v_fixed = np.linalg.solve(lin, v_world.T).T
    v_vox = v_fixed @ _get_rotation(nib.load(in_fa).affine)
    v_vox /= np.linalg.norm(v_vox, axis=-1, keepdims=True)
    return v_vox * AMPLITUDES[:, None]


def _expected_warp_directions(in_warp):
    """
    Unit peak directions expected in the fa voxel frame with the warp, using
    the inverse warp v directly. The moving -> fixed mapping is
    psi(y) = z + v(z), with z = inv(A)(y), so the inverse of the
    fixed -> moving Jacobian is J_psi = (I + grad v(z)) @ inv(L), where
    z = x + u(x) for each fixed point x.
    """
    fa = nib.load(in_fa)
    lin = load_matrix_in_any_format(in_aff)[:3, :3]
    v, v_vox2ras = load_ants_warp(in_inv_warp)
    u, u_vox2ras = load_ants_warp(in_warp)
    # bst: the warps are defined on the fa grid.
    np.testing.assert_allclose(u_vox2ras, fa.affine)
    np.testing.assert_allclose(v_vox2ras, fa.affine)

    # d v_ras / d x_ras on the warp grid, (X,Y,Z,3,3).
    grad_v = np.stack(np.gradient(v, axis=(0, 1, 2)), axis=-1) \
        @ np.linalg.inv(v_vox2ras[:3, :3])
    grid = np.indices(fa.shape).transpose(1, 2, 3, 0).astype(np.float64)
    z = apply_affine(fa.affine, grid) + u
    coords = apply_affine(np.linalg.inv(v_vox2ras), z).reshape(-1, 3).T
    grad_v_z = np.stack([map_coordinates(grad_v[..., i, j], coords, order=1,
                                         mode='nearest')
                         for i in range(3) for j in range(3)], axis=-1)
    inv_jac = (np.eye(3) + grad_v_z.reshape(fa.shape + (3, 3))) \
        @ np.linalg.inv(lin)

    v_world = PEAKS_VOX @ _get_rotation(nib.load(in_template).affine).T
    v_vox = np.einsum('...ij,pj->...pi', inv_jac, v_world) \
        @ _get_rotation(fa.affine)
    return v_vox / np.linalg.norm(v_vox, axis=-1, keepdims=True)


def test_help_option(script_runner):
    ret = script_runner.run(['scil_peak_apply_transform', '--help'])
    assert ret.success


def test_execution_identity(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    rng = np.random.default_rng(0)
    fa = nib.load(in_fa)
    peaks = rng.normal(size=fa.shape[:3] + (6,)).astype(np.float32)
    nib.save(nib.Nifti1Image(peaks, fa.affine), 'peaks_fa.nii.gz')
    np.savetxt('identity.txt', np.eye(4))

    ret = script_runner.run(['scil_peak_apply_transform', 'peaks_fa.nii.gz',
                             in_fa, 'identity.txt', 'peaks_identity.nii.gz'])
    assert ret.success

    out = nib.load('peaks_identity.nii.gz')
    np.testing.assert_allclose(out.affine, fa.affine)
    np.testing.assert_allclose(out.get_fdata(), peaks, atol=1e-5)


def test_execution_affine(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    _save_constant_peaks(in_template, 'peaks_template.nii.gz')

    ret = script_runner.run(['scil_peak_apply_transform',
                             'peaks_template.nii.gz', in_fa, in_aff,
                             'peaks_affine.nii.gz'])
    assert ret.success

    out = nib.load('peaks_affine.nii.gz')
    assert out.shape == nib.load(in_fa).shape[:3] + (6,)
    data = out.get_fdata().reshape(out.shape[:3] + (2, 3))
    inside = np.any(data != 0, axis=(-2, -1))
    assert inside.sum() > 0.5 * inside.size

    # An affine transform reorients every peak the same way.
    expected = _expected_affine_peaks()
    np.testing.assert_allclose(data[inside],
                               np.broadcast_to(expected, data[inside].shape),
                               atol=1e-4)


def test_execution_warp(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    _save_constant_peaks(in_template, 'peaks_template.nii.gz')
    _build_forward_warp(in_inv_warp, 'output1Warp.nii.gz')

    ret = script_runner.run(['scil_peak_apply_transform',
                             'peaks_template.nii.gz', in_fa, in_aff,
                             'peaks_warp.nii.gz',
                             '--warp', 'output1Warp.nii.gz',
                             '--mask', in_mask])
    assert ret.success

    out = nib.load('peaks_warp.nii.gz')
    assert out.shape == nib.load(in_fa).shape[:3] + (6,)
    data = out.get_fdata().reshape(out.shape[:3] + (2, 3))
    inside = np.any(data != 0, axis=(-2, -1))
    mask = nib.load(in_mask).get_fdata() > 0
    assert not np.any(inside & ~mask)
    assert inside.sum() > 0.5 * mask.sum()

    # Amplitudes are preserved.
    np.testing.assert_allclose(np.linalg.norm(data[inside], axis=-1),
                               np.broadcast_to(AMPLITUDES,
                                               (inside.sum(), 2)),
                               rtol=1e-4)

    # Directions match the Jacobian derived independently from the inverse
    # warp (not from the finite differences used by the script).
    expected = _expected_warp_directions('output1Warp.nii.gz')[inside]
    cos = np.sum(data[inside] / AMPLITUDES[:, None] * expected, axis=-1)
    angles = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    assert np.median(angles) < 0.5
    assert np.percentile(angles, 95) < 1
    assert angles.max() < 5
