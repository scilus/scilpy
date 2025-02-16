#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize collisions given by scil_tractogram_filter_collisions with
the --save_colliding parameter.
"""

import argparse

from dipy.io.streamline import load_tractogram
from scilpy.io.streamlines import load_tractogram_with_reference
from fury import window, actor
from nibabel.streamlines import detect_format, TrkFile

from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_tractogram_invalid',
                   help='Tractogram file containing the colliding \n'
                   'streamlines that have been filtered, along their \n'
                   'collision point as data_per_streamline (must be \n'
                   '.trk). This file is obtained from the \n'
                   'scil_tractogram_filter_collisions.py script.')

    p.add_argument('--in_tractogram_obstacle',
                   help='Tractogram file containing the streamlines that \n'
                   'that [in_tractogram_invalid] has collided with. Will \n'
                   'be overlaid in the viewing window. This file is \n'
                   'obtained from the scil_tractogram_filter_collisions.py \n'
                   'script.')

    p.add_argument('--ref_tractogram',
                   help='Tractogram file containing the full tractogram \n'
                   'as visual reference (must be .trk or .tck). It will be \n'
                   'overlaid in white and very low opacity.')

    p.add_argument('--out_screenshot', default='',
                   help='If set, save a screenshot of the result in the \n'
                   'specified filename (.png, .bmp, .jpeg or .jpg).')

    p.add_argument('--win_size', nargs=2, type=int, default=(1000, 1000))

    add_overwrite_arg(p)
    add_reference_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram_invalid,
                        [args.in_tractogram_obstacle, args.ref_tractogram])
    assert_outputs_exist(parser, args, [], [args.out_screenshot])

    tracts_format = detect_format(args.in_tractogram_invalid)
    if tracts_format is not TrkFile:
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk):" +
                         "{0}".format(args.in_tractogram_invalid))

    if args.in_tractogram_obstacle:
        tracts_format = detect_format(args.in_tractogram_obstacle)
        if tracts_format is not TrkFile:
            raise ValueError("Invalid input streamline file format " +
                             "(must be trk):" +
                             "{0}".format(args.in_tractogram_invalid))

    invalid_sft = load_tractogram(args.in_tractogram_invalid, 'same',
                                  bbox_valid_check=False)
    invalid_sft.to_voxmm()
    invalid_sft.to_center()

    if 'collisions' not in invalid_sft.data_per_streamline:
        parser.error('Tractogram does not contain collisions')
    collisions = invalid_sft.data_per_streamline['collisions']

    if args.in_tractogram_obstacle:
        obstacle_sft = load_tractogram(args.in_tractogram_obstacle, 'same',
                                       bbox_valid_check=False)
        obstacle_sft.to_voxmm()
        obstacle_sft.to_center()
    if args.ref_tractogram:
        full_sft = load_tractogram_with_reference(parser, args,
                                                  args.ref_tractogram)
        full_sft.to_voxmm()
        full_sft.to_center()

    # Make display objects and add them to canvas
    s = window.Scene()
    invalid_actor = actor.line(invalid_sft.streamlines,
                               colors=[1., 0., 0.])
    s.add(invalid_actor)

    if args.in_tractogram_obstacle:
        obstacle_actor = actor.line(obstacle_sft.streamlines,
                                    colors=[0., 1., 0.])
        s.add(obstacle_actor)

    if args.ref_tractogram:
        full_actor = actor.line(full_sft.streamlines, opacity=0.03,
                                colors=[1., 1., 1.])

        s.add(full_actor)

    points = actor.dot(collisions, colors=(1., 1., 1.))
    s.add(points)

    # Show and record if needed
    if args.out_screenshot:
        window.record(s, out_path=args.out_screenshot, size=args.win_size)
    window.show(s)


if __name__ == '__main__':
    main()
