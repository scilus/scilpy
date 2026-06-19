#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import argparse
from tqdm import tqdm

from scilpy.io.utils import add_overwrite_arg, assert_inputs_exist, assert_outputs_exist


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Concatenate multiple volumes into a single 4D volume.")
    p.add_argument('out_volume', help='Output concatenated volume.')
    p.add_argument('in_volumes', nargs='+', help='Input volumes to concatenate (must be in the same space).')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_volumes)
    assert_outputs_exist(parser, args, args.out_volume)

    ref = nib.load(args.in_volumes[0])
    concat = np.zeros(ref.shape + (len(args.in_volumes),),
                      dtype=ref.get_data_dtype())

    for idx, vol in enumerate(tqdm(args.in_volumes)):
        if not nib.load(vol).shape == nib.load(args.in_volumes[0]).shape:
            parser.error(f"All input volumes must have the same shape. {vol} has a different shape than {args.in_volumes[0]}.")
        im = nib.load(vol)
        arr = np.asarray(im.dataobj)
        concat[..., idx] = arr

    nib.save(nib.Nifti1Image(concat, ref.affine),
             args.out_volume)


if __name__ == "__main__":
    main()