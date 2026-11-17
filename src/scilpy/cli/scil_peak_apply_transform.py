#!/usr/bin/env python
"""Warp fODF peaks into MNI space using an ANTs affine + SyN warp.

The ANTs registration was run with fixed = MNI and moving = subject, so
resampling the peaks into MNI corresponds to:

    antsApplyTransforms -r MNI -t Warp.nii.gz -t GenericAffine.mat

ANTs/ITK transforms are expressed in LPS physical space; they are converted
to RAS when loaded so that all computations use the nibabel (RAS) affines.

Peaks are resampled with nearest-neighbour interpolation (interpolating peaks
of arbitrary sign/order is meaningless), and each peak vector is reoriented by
the inverse local Jacobian of the fixed->moving mapping, keeping its amplitude.
"""
import argparse
import logging

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from scipy.ndimage import map_coordinates

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (
    add_overwrite_arg,
    add_verbose_arg,
    assert_headers_compatible,
    assert_inputs_exist,
    assert_outputs_exist,
    load_ants_warp,
    load_matrix_in_any_format,
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_peaks",
                   help="Peaks image (X,Y,Z,3*N) to transform.")
    p.add_argument("in_reference",
                   help="Reference (fixed) image defining the output grid.")
    p.add_argument("in_affine",
                   help="Affine transform (fixed -> moving). Either an ANTs\n"
                        ".mat file (LPS, converted to RAS) or a .txt/.npy\n"
                        "4x4 matrix already in RAS.")
    p.add_argument("out_peaks", help="Output transformed peaks image.")
    p.add_argument("--warp", help="Optional ANTs warp image for non-linear transform.")
    p.add_argument("--mask", help="Optional mask in reference space.")
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def map_fixed_to_moving(pts, affine, warp=None):
    """
    Fixed -> moving RAS point mapping: warp first, then affine. As per
    ANTs documentation, to move points from fixed to moving spaces, the
    forward transforms should be used.

    Parameters
    ----------
    pts: np.ndarray
        Input points array.
    affine: np.ndarray
        The transformation affine matrix.
    warp: tuple, optional
        Displacement field (disp, warp_vox2ras). The warp describes where a voxel
        in reference space lands in the moving space. Defaults to None.
    Returns
    -------
    np.ndarray: The mapped points array.
    """
    if warp is not None:
        disp, warp_vox2ras = warp
        idx = apply_affine(np.linalg.inv(warp_vox2ras), pts)
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


def get_rotation(affine):
    """
    Rotation (direction cosines) of a voxel to world affine. Because the Nifti
    voxel-to-world affine has the form `M = Rot @ diag(zooms)`, the norm of
    each column corresponds to `zooms`. So, we can extract the rotation
    component by dividing by norm(M, axis=0).

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

    assert_inputs_exist(parser, [args.in_peaks, args.in_reference, args.in_affine],
                        [args.warp, args.mask])
    assert_outputs_exist(parser, args, args.out_peaks)
    assert_headers_compatible(parser, args.in_reference, [args.warp, args.mask])

    peaks_nib = nib.load(args.in_peaks)
    ref_nib = nib.load(args.in_reference)

    peaks_moving = peaks_nib.get_fdata()

    # Safety check to ensure the input is a peaks image
    if peaks_nib.ndim != 4 or peaks_nib.shape[3] % 3:
        raise ValueError(
            f"Peaks image must be 4D with a last dimension divisible by 3; got shape {peaks_nib.shape}.")
    n_peaks = peaks_nib.shape[3] // 3

    affine = load_matrix_in_any_format(args.in_affine)
    warp = load_ants_warp(args.warp) if args.warp else None

    # Build a reference grid.
    ref_shape = ref_nib.shape[:3]
    ref_grid = np.stack(np.meshgrid(*[np.arange(s) for s in ref_shape],
                                indexing="ij"), axis=-1).astype(np.float64)
    # Get positions in moving image (in world space).
    mapped = map_fixed_to_moving(apply_affine(ref_nib.affine, ref_grid), affine, warp)

    # Inverse affine mapping moving world coordinates back to voxel indices.
    inv_affine = np.linalg.inv(peaks_nib.affine)

    # Calculate moving voxel indices by rounding them to the nearest integer.
    logger.info("Resampling peaks (nearest neighbour)...")
    idx = np.rint(apply_affine(inv_affine, mapped)).astype(np.int64)

    # Mask voxels that are outside the moving voxel grid.
    is_valid = np.all((idx >= 0) & (idx < np.array(peaks_moving.shape[:3])), axis=-1)

    # Create output peaks array and fill with corresponding moving peaks.
    out_peaks = np.zeros(ref_shape + (peaks_nib.shape[3],), dtype=np.float64)
    out_peaks[is_valid] = peaks_moving[tuple(idx[is_valid].T)]

    nonzero_peaks = np.ones(ref_shape, dtype=bool)
    # Optionally mask out_peaks that are zero in the reference volume.
    if args.mask is not None:
        ref_mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
        nonzero_peaks = ref_mask
    # Mask containing the non-zero voxels in out_peaks.
    nonzero_peaks &= np.any(out_peaks != 0, axis=-1)

    logger.info("Computing Jacobian and reorienting peaks...")

    # Gradient of the warped voxel grid: moving mm per fixed voxels
    # This is d moving_mm / d fixed_vox. This is a (nx, ny, nz, 3, 3) matrix.
    displacement_mm_per_vox = np.stack(np.gradient(mapped, axis=(0, 1, 2)), axis=-1)

    # We want moving mm per fixed mm: d moving_mm / d fixed_mm.
    # Because fixed_vox = inv(ref_nib.affine) * fixed_mm, we get
    # d fixed_vox / d fixed_mm = inv(ref_nib.affine), and, by chain rule,
    # d moving_mm / d fixed_mm = d moving_mm / d fixed_vox * d fixed_vox /d fixed_mm
    jac = displacement_mm_per_vox[nonzero_peaks] @ np.linalg.inv(ref_nib.affine[:3, :3])  # (N,3,3)

    # Peaks voxel frame -> RAS world (moving), and RAS world (fixed) -> reference voxel frame
    peaks_to_world = get_rotation(peaks_nib.affine)
    # for a rotation matrix, the transpose is the inverse
    world_to_ref = get_rotation(ref_nib.affine).T

    # Compute the new peak positions in reference space
    vecs = out_peaks[nonzero_peaks].reshape(-1, n_peaks, 3)  # (N,P,3)
    v_ras = vecs @ peaks_to_world.T  # moving RAS
    # multiply by the inverse of the jacobian to get the fixed mm
    v_fixed_mm = np.linalg.solve(jac, v_ras.transpose(0, 2, 1)).transpose(0, 2, 1)
    # finally transform to fixed voxels
    v_out_vox = v_fixed_mm @ world_to_ref.T

    # Normalize peaks so they keep the same length as before
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    new_norms = np.linalg.norm(v_out_vox, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_out_vox = np.where(new_norms > 0, v_out_vox * norms / new_norms, 0.0)
    out_peaks[nonzero_peaks] = v_out_vox.reshape(-1, peaks_nib.shape[3])

    out_peaks[~nonzero_peaks] = 0
    nib.save(nib.Nifti1Image(out_peaks.astype(np.float32), ref_nib.affine), args.out_peaks)
    logger.info(f"Saved {args.out_peaks} {out_peaks.shape}")


if __name__ == "__main__":
    main()
