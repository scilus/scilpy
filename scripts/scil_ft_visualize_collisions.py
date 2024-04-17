#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize collisions found through the intersection filtering process of
scil_ft_filter_collisions.py (with the --save_colliders option).

The obtained "colliders" tractogram is to be used as input in the present
script. The collision points need to be stored as data_per_streamline on
each of the streamlines.
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


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('colliders',
                   help='Tractogram file containing the colliding \n'
                   'streamlines and their collision points (must be .trk). \n')

    p.add_argument('--collided',
                   help='Tractogram file containing the streamlines that \n'
                   'have been collided with (must be .trk). Will be \n'
                   'overlaid in the viewing window.')

    p.add_argument('--ref_tractogram',
                   help='Tractogram file containing the full tractogram \n'
                   'as visual reference (must be .trk or .tck). It will be'
                   'overlaid in white and very low opacity.')

    p.add_argument('--save',
                   help='If set, save a screenshot of the result in the \n'
                   'specified filename (.png, .bmp, .jpeg or .jpg).')

    add_overwrite_arg(p)
    add_reference_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.colliders,
                        [args.collided, args.ref_tractogram])
    assert_outputs_exist(parser, args, [], [args.save])

    tracts_format = detect_format(args.colliders)
    if tracts_format is not TrkFile:
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk): {0}".format(args.colliders))

    if args.collided:
        tracts_format = detect_format(args.collided)
        if tracts_format is not TrkFile:
            raise ValueError("Invalid input streamline file format " +
                             "(must be trk): {0}".format(args.colliders))

    colliders_sft = load_tractogram(args.colliders, 'same',
                                    bbox_valid_check=False)
    colliders_sft.to_voxmm()
    colliders_sft.to_center()

    if 'collisions' not in colliders_sft.data_per_streamline:
        parser.error('Tractogram does not contain collisions')
    collisions = colliders_sft.data_per_streamline['collisions']

    if (args.collided):
        collided_sft = load_tractogram(args.collided, 'same',
                                       bbox_valid_check=False)
        collided_sft.to_voxmm()
        collided_sft.to_center()
    if (args.ref_tractogram):
        full_sft = load_tractogram_with_reference(parser, args,
                                                  args.ref_tractogram)
        full_sft.to_voxmm()
        full_sft.to_center()

    # Make display objects and add them to canvas
    s = window.Scene()
    colliders_actor = actor.line(colliders_sft.streamlines,
                                 colors=[1., 0., 0.])
    s.add(colliders_actor)

    if (args.collided):
        collided_actor = actor.line(collided_sft.streamlines,
                                    colors=[0., 1., 0.])
        s.add(collided_actor)

    if (args.ref_tractogram):
        full_actor = actor.line(full_sft.streamlines, opacity=0.03,
                                colors=[1., 1., 1.])

        s.add(full_actor)

    points = actor.dot(collisions, colors=(1., 1., 1.))
    s.add(points)

    # Show and record if needed
    if args.save is not None:
        window.record(s, out_path=args.save, size=(1000, 1000))
    window.show(s)


if __name__ == '__main__':
    main()
