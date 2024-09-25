#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize seeds used to generate the tractogram or bundle.
When tractography was run, each streamline produced by the tracking algorithm
saved its seeding point (its origin).

The tractogram must have been generated from scil_tracking_local.py or
scil_tracking_pft.py with the --save_seeds option.
"""

import argparse
import logging

from dipy.io.streamline import load_tractogram
from fury import window, actor
from nibabel.streamlines import detect_format, TrkFile

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('tractogram',
                   help='Tractogram file (must be trk)')
    p.add_argument('--save',
                   help='If set, save a screenshot of the result in the '
                        'specified filename')

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.tractogram])
    assert_outputs_exist(parser, args, [], [args.save])

    tracts_format = detect_format(args.tractogram)
    if tracts_format is not TrkFile:
        raise ValueError("Invalid input streamline file format " +
                         "(must be trk): {0}".format(args.tractogram_filename))

    # Load files and data. TRKs can have 'same' as reference
    tractogram = load_tractogram(args.tractogram, 'same')
    # Streamlines are saved in RASMM but seeds are saved in VOX
    # This might produce weird behavior with non-iso
    tractogram.to_vox()

    streamlines = tractogram.streamlines
    if 'seeds' not in tractogram.data_per_streamline:
        parser.error('Tractogram does not contain seeds')
    seeds = tractogram.data_per_streamline['seeds']

    # Make display objects
    streamlines_actor = actor.line(streamlines)
    points = actor.dot(seeds, color=(1., 1., 1.))

    # Add display objects to canvas
    s = window.Scene()
    s.add(streamlines_actor)
    s.add(points)

    # Show and record if needed
    if args.save is not None:
        window.record(s, out_path=args.save, size=(1000, 1000))
    window.show(s)


if __name__ == '__main__':
    main()
