#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

from dipy.io.streamline import save_tractogram

from scilpy.io.utils import (add_overwrite_arg, add_reference,
                             assert_inputs_exist, assert_outputs_exists,
                             load_tractogram_with_reference)

DESCRIPTION = """
Removal of streamlines that are out of the volume bounding box. In voxel space
no negative coordinate and no above volume dimension coordinate are possible.
Any streamlines that do not respect these two conditions are removed.
"""


def _build_args_parser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram', metavar='IN_TRACTOGRAM',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy')

    p.add_argument('output_name', metavar='OUTPUT_NAME',
                   help='Output filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy')

    add_reference(p)

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_tractogram], [args.reference])

    in_extension = os.path.splitext(args.in_tractogram)[1]
    out_extension = os.path.splitext(args.output_name)[1]

    assert_outputs_exists(parser, args, args.output_name)

    sft = load_tractogram_with_reference(parser, args, args.in_tractogram,
                                         bbox_check=False)
    sft.remove_invalid_streamlines()
    save_tractogram(sft, args.output_name)


if __name__ == "__main__":
    main()
