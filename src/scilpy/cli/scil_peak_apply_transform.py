#!/usr/bin/env python
"""Warp fODF peaks into MNI space using an ANTs affine + SyN warp.

The ANTs registration was run with fixed = MNI and moving = subject, so
resampling the peaks into MNI corresponds to:

    antsApplyTransforms -r MNI -t Warp.nii.gz -t GenericAffine.mat

Peaks are resampled with nearest-neighbour interpolation (interpolating peaks
of arbitrary sign/order is meaningless), and each peak vector is reoriented by
the inverse local Jacobian of the fixed->moving mapping, keeping its amplitude.
"""
import argparse
import logging

import nibabel as nib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import map_coordinates

from scilpy.io.utils import (
    add_overwrite_arg,
    add_verbose_arg,
    assert_inputs_exist,
    assert_outputs_exist,
)

# ITK/ANTs work in LPS physical space, nibabel in RAS.
RAS2LPS = np.diag([-1.0, -1.0, 1.0])


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_peaks",
                   help="Peaks image (X,Y,Z,3*N) to transform.")
    p.add_argument("in_reference",
                   help="Reference (fixed) image defining the output grid.")
    p.add_argument("in_affine", help="ANTs affine matrix for linear transform.")
    p.add_argument("out_peaks", help="Output transformed peaks image.")
    p.add_argument("--warp", help="Optional ANTs warp image for non-linear transform.")
    p.add_argument("--mask", action="store_true",
                   help="Mask voxels that are 0 in the reference image.")
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def vox2lps(affine):
    """
    Voxel index -> LPS physical point affine (4x4) from a RAS nifti affine.

    Parameters
    ----------
    affine: np.ndarray
        The RAS nifti affine matrix.
    Returns
    -------
    np.ndarray: The 4x4 LPS physical point affine matrix.
    """
    out = np.eye(4)
    out[:3] = RAS2LPS @ affine[:3]
    return out


def load_ants_affine(path):
    """
    Read an ANTs .mat affine as a 4x4 matrix mapping LPS points
    fixed -> moving.

    Parameters
    ----------
    path: str
        Path to the ANTs .mat affine file.

    Returns
    -------
    np.ndarray: The 4x4 LPS affine matrix.
    """
    mat = loadmat(path)
    key = next(k for k in mat if k.startswith("AffineTransform_"))
    params = np.asarray(mat[key], dtype=np.float64).ravel()
    a = params[:9].reshape(3, 3)
    t = params[9:12]
    c = np.asarray(mat["fixed"], dtype=np.float64).ravel()
    out = np.eye(4)
    out[:3, :3] = a
    out[:3, 3] = t + c - a @ c
    return out


def load_ants_warp(path):
    """
    Read an ANTs displacement field.

    Parameters
    ----------
    path: str
        Path to the ANTs warp image.

    Returns
    -------
    tuple: A tuple containing the (X,Y,Z,3) displacement array and the voxel -> LPS affine matrix.
    """
    disp_img = nib.load(path)
    displacement = np.asarray(disp_img.dataobj, dtype=np.float64).reshape(disp_img.shape[:3] + (3,))
    return displacement, vox2lps(disp_img.affine)


def apply_affine(aff, pts):
    """
    Apply a 4x4 affine to points of shape (...,3).

    Parameters
    ----------
    aff: np.ndarray
        The 4x4 affine matrix.
    pts: np.ndarray
        Array of points with shape (..., 3).

    Returns
    -------
    np.ndarray: The transformed points array.
    """
    return pts @ aff[:3, :3].T + aff[:3, 3]


def map_points(pts, affine, warp=None):
    """
    Fixed -> moving LPS point mapping: inverse warp first, then affine.

    Parameters
    ----------
    pts: np.ndarray
        Input points array.
    affine: np.ndarray
        The transformation affine matrix.
    warp: tuple, optional
        Displacement field (disp, warp_vox2lps). Defaults to None.
    Returns
    -------
    np.ndarray: The mapped points array.
    """
    if warp is not None:
        disp, warp_vox2lps = warp
        idx = apply_affine(np.linalg.inv(warp_vox2lps), pts)
        shape = np.array(disp.shape[:3])
        coords = idx.reshape(-1, 3).T
        d = np.stack([map_coordinates(disp[..., k], coords, order=1,
                                      mode="nearest")
                      for k in range(3)], axis=-1)
        # Like ITK, no displacement outside the field.
        outside = np.any((idx.reshape(-1, 3) < -0.5) |
                         (idx.reshape(-1, 3) > shape - 0.5), axis=-1)
        d[outside] = 0
        pts = pts + d.reshape(pts.shape)
    return apply_affine(affine, pts)


def rotation(affine):
    """
    Rotation (direction cosines) of a voxel -> world affine.

    Parameters
    ----------
    affine: np.ndarray
        Voxel->world affine matrix.

    Returns
    -------
    np.ndarray: The normalized rotation matrix (direction cosines).
    """
    m = affine[:3, :3]
    return m / np.linalg.norm(m, axis=0)


def main():
    parser = build_parser()
    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_peaks, args.in_reference, args.in_affine], args.warp)
    assert_outputs_exist(parser, args, args.out_peaks)

    peaks_nib = nib.load(args.in_peaks)
    ref_nib = nib.load(args.in_reference)
    n_comp = peaks_nib.shape[3]
    if n_comp % 3:
        raise ValueError(f"Last dimension ({n_comp}) is not a multiple of 3.")
    n_peaks = n_comp // 3

    affine = load_ants_affine(args.in_affine)
    warp = load_ants_warp(args.warp) if args.warp else None

    # Physical (LPS) position of every reference voxel, mapped to the moving space.
    ref_vox2lps = vox2lps(ref_nib.affine)
    ref_shape = ref_nib.shape[:3]
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in ref_shape],
                                indexing="ij"), axis=-1).astype(np.float64)
    mapped = map_points(apply_affine(ref_vox2lps, grid), affine, warp)

    logger.info("Resampling peaks (nearest neighbour)...")
    peaks = np.asarray(peaks_nib.dataobj, dtype=np.float32)

    # Get the inverse affine mapping from LPS physical coordinates back to voxel indices.
    inv_affine = np.linalg.inv(vox2lps(peaks_nib.affine))
    # Calculate floating-point voxel indices and round them to the nearest integer.
    idx = np.rint(apply_affine(inv_affine, mapped)).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.array(peaks.shape[:3])), axis=-1)
    out = np.zeros(ref_shape + (n_comp,), dtype=np.float64)
    out[inside] = peaks[tuple(idx[inside].T)]

    mask = np.ones(ref_shape, dtype=bool)
    if args.mask:
        mask = np.asarray(ref_nib.dataobj) > 0
    mask &= np.any(out != 0, axis=-1)

    logger.info("Computing Jacobian and reorienting peaks...")
    # Jacobian of the fixed->moving mapping (LPS), from finite differences
    # with respect to voxel indices, converted to physical units.
    d_index = np.stack(np.gradient(mapped, axis=(0, 1, 2)), axis=-1)
    jac = d_index[mask] @ np.linalg.inv(ref_vox2lps[:3, :3])  # (N,3,3)

    # Peaks voxel frame -> LPS world (moving), and LPS world (fixed) -> MNI voxel frame
    to_lps = RAS2LPS @ rotation(peaks_nib.affine)
    from_lps = rotation(ref_nib.affine).T @ RAS2LPS

    vecs = out[mask].reshape(-1, n_peaks, 3)  # (N,P,3)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    v_lps = vecs @ to_lps.T  # moving LPS
    v_fix = np.linalg.solve(jac, v_lps.transpose(0, 2, 1)).transpose(0, 2, 1)
    v_out = v_fix @ from_lps.T
    new_norms = np.linalg.norm(v_out, axis=-1, keepdims=True)
    # suppress divisions by 0 (these should not happen thanks to `np.where`,
    # but the warning may still be raised)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_out = np.where(new_norms > 0, v_out * norms / new_norms, 0.0)
    out[mask] = v_out.reshape(-1, n_comp)

    out[~mask] = 0
    nib.save(nib.Nifti1Image(out.astype(np.float32), ref_nib.affine), args.out_peaks)
    logger.info(f"Saved {args.out_peaks} {out.shape}")


if __name__ == "__main__":
    main()
