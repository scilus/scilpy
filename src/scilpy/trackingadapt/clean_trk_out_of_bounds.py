#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import nibabel as nib

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import Space, Origin, StatefulTractogram


def filter_out_of_bounds_sft(sft, ref_img):
    """Keep only streamlines fully inside [0, shape) in VOX/CORNER space."""
    img = nib.load(ref_img)
    shape = img.shape[:3]

    # Ensure we are comparing in voxel indices
    sft.to_space(Space.VOX)
    sft.to_origin(Origin("corner"))

    kept = []
    for sl in sft.streamlines:
        pts = np.asarray(sl)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            continue
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        inside = (
            (x >= 0) & (x < shape[0]) &
            (y >= 0) & (y < shape[1]) &
            (z >= 0) & (z < shape[2])
        )
        if bool(np.all(inside)):
            kept.append(sl)

    return kept, img


def main():
    p = argparse.ArgumentParser(
        description="Remove streamlines with out-of-bounds voxel coordinates."
    )
    p.add_argument("in_trk", help="Input tractogram (.trk/.tck)")
    p.add_argument("ref_img", help="Reference NIfTI used to interpret tractogram space (e.g. wm.nii.gz)")
    p.add_argument("out_trk", help="Output cleaned tractogram (.trk/.tck)")
    p.add_argument("--keep-empty", action="store_true",
                   help="If set, save even if 0 streamlines remain.")
    args = p.parse_args()

    sft = load_tractogram(args.in_trk, args.ref_img, bbox_valid_check=False)

    kept, img = filter_out_of_bounds_sft(sft, args.ref_img)

    print(f"Keeping {len(kept)}/{len(sft.streamlines)} streamlines")

    if len(kept) == 0 and not args.keep_empty:
        raise SystemExit("0 streamlines kept. Use --keep-empty to still write an output file.")

    sft_clean = StatefulTractogram(
        kept, img, space=Space.VOX, origin=Origin("corner")
    )

    save_tractogram(sft_clean, args.out_trk, bbox_valid_check=False)
    print("Saved:", args.out_trk)


if __name__ == "__main__":
    main()
