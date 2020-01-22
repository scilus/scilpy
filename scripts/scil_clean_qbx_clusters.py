#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import logging

from fury import window, actor, interactor
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty)

DESCRIPTION = """
    Render clusters sequentially to either accept or reject them based on
    visual inspection. Useful for cleaning bundles for RBx, BST or for figures.
    The VTK window does not handle well opacity of streamlines, this is a normal
    rendering behavior.
    Often use in pair with scil_compute_qbx.py.

    Key mapping:
    - a/A: accept displayed clusters
    - r/R: reject displayed clusters
    - z/Z: Rewing one element
    - c/C: Stop rendering of the background concatenation of streamlines
    - q/Q: Early window exist, everything remaining will be rejected
"""


def _build_args_parser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundles', nargs='+',
                   help='List of the clusters filename.')
    p.add_argument('out_accepted',
                   help='Filename of the concatenated accepted clusters.')
    p.add_argument('out_rejected',
                   help='Filename of the concatenated rejected clusters.')

    p.add_argument('--out_accepted_dir',
                   help='Directory to save all accepted clusters separately.')
    p.add_argument('--out_rejected_dir',
                   help='Directory to save all rejected clusters separately.')

    p.add_argument('--min_cluster_size', type=int, default=1,
                   help='Minimum cluster size for consideration [%(default)s].'
                        'Must be at least 1.')
    p.add_argument('--background_opacity', type=float, default=0.1,
                   help='Opacity of the background streamlines.'
                        'Keep low between 0 and 0.5 [%(default)s].')
    p.add_argument('--background_linewidth', type=float, default=1,
                   help='Linewidth of the background streamlines [%(default)s].')
    p.add_argument('--clusters_linewidth', type=float, default=1,
                   help='Linewidth of the current cluster [%(default)s].')

    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def get_length_tuple(elem):
    return len(elem[0])


def main():
    # Callback required for FURY
    def keypress_callback(obj, _):
        key = obj.GetKeySym().lower()
        nonlocal clusters_linewidth, background_linewidth
        nonlocal curr_streamlines_actor, concat_streamlines_actor, show_curr_actor
        iterator = len(accepted_streamlines) + len(rejected_streamlines)
        renwin = interactor_style.GetInteractor().GetRenderWindow()
        renderer = interactor_style.GetCurrentRenderer()

        if key == 'c' and iterator < len(sft_accepted_on_size):
            if show_curr_actor:
                renderer.rm(concat_streamlines_actor)
                renwin.Render()
                show_curr_actor = False
                logging.info('Streamlines rendering OFF')
            else:
                renderer.add(concat_streamlines_actor)
                renderer.rm(curr_streamlines_actor)
                renderer.add(curr_streamlines_actor)
                renwin.Render()
                show_curr_actor = True
                logging.info('Streamlines rendering ON')
            return

        if key == 'q':
            show_manager.exit()
            if iterator < len(sft_accepted_on_size):
                logging.warning(
                    'Early exit, everything remaining to be rejected.')
            return

        if key in ['a', 'r'] and iterator < len(sft_accepted_on_size):
            if key == 'a':
                accepted_streamlines.append(iterator)
                choices.append('a')
                logging.info('Accepted file %s',
                             filename_accepted_on_size[iterator])
            elif key == 'r':
                rejected_streamlines.append(iterator)
                choices.append('r')
                logging.info('Rejected file %s',
                             filename_accepted_on_size[iterator])
            iterator += 1

        if key == 'z':
            if iterator > 0:
                last_choice = choices.pop()
                if last_choice == 'r':
                    rejected_streamlines.pop()
                else:
                    accepted_streamlines.pop()
                logging.info('Rewind on step.')

                iterator -= 1
            else:
                logging.warning('Cannot rewind, first element.')

        if key in ['a', 'r', 'z'] and iterator < len(sft_accepted_on_size):
            renderer.rm(curr_streamlines_actor)
            curr_streamlines = sft_accepted_on_size[iterator].streamlines
            curr_streamlines_actor = actor.line(curr_streamlines,
                                                opacity=0.8,
                                                linewidth=clusters_linewidth)
            renderer.add(curr_streamlines_actor)

        if iterator == len(sft_accepted_on_size):
            print('No more cluster, press q to exit')
            renderer.rm(curr_streamlines_actor)

        renwin.Render()

    parser = _build_args_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_bundles)
    assert_outputs_exist(parser, args, [args.out_accepted, args.out_rejected])

    if args.out_accepted_dir:
        assert_output_dirs_exist_and_empty(parser, args,
                                           args.out_accepted_dir,
                                           create_dir=True)
    if args.out_rejected_dir:
        assert_output_dirs_exist_and_empty(parser, args,
                                           args.out_rejected_dir,
                                           create_dir=True)

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.min_cluster_size < 1:
        parser.error('Minimum cluster size must be at least 1.')

    clusters_linewidth = args.clusters_linewidth
    background_linewidth = args.background_linewidth

    # To accelerate procedure, clusters can be discarded based on size
    # Concatenation is to give spatial context
    sft_accepted_on_size, filename_accepted_on_size = [], []
    sft_rejected_on_size, filename_rejected_on_size = [], []
    concat_streamlines = []
    for filename in args.in_bundles:
        if not is_header_compatible(args.in_bundles[0], filename):
            return
        basename = os.path.basename(filename)
        sft = load_tractogram_with_reference(parser, args, filename,
                                             bbox_check=False)
        if len(sft) >= args.min_cluster_size:
            sft_accepted_on_size.append(sft)
            filename_accepted_on_size.append(basename)
            concat_streamlines.extend(sft.streamlines)
        else:
            logging.info('File %s has %s streamlines, automatically rejected.',
                         filename, len(sft))
            sft_rejected_on_size.append(sft)
            filename_rejected_on_size.append(basename)

    logging.info('%s clusters to be classified.', len(sft_accepted_on_size))
    # The clusters are sorted by size for simplicity/efficiency
    tuple_accepted = zip(*sorted(zip(sft_accepted_on_size,
                                     filename_accepted_on_size),
                                 key=get_length_tuple, reverse=True))
    sft_accepted_on_size, filename_accepted_on_size = tuple_accepted

    # Initialize the actors, scene, window, observer
    concat_streamlines_actor = actor.line(concat_streamlines,
                                          colors=(1, 1, 1),
                                          opacity=args.background_opacity,
                                          linewidth=background_linewidth)
    curr_streamlines_actor = actor.line(sft_accepted_on_size[0].streamlines,
                                        opacity=0.8,
                                        linewidth=clusters_linewidth)

    scene = window.Scene()
    interactor_style = interactor.CustomInteractorStyle()
    show_manager = window.ShowManager(scene, size=(800, 800),
                                      reset_camera=False,
                                      interactor_style=interactor_style)
    scene.add(concat_streamlines_actor)
    scene.add(curr_streamlines_actor)
    interactor_style.AddObserver('KeyPressEvent', keypress_callback)

    # Lauch rendering and selection procedure
    choices, accepted_streamlines, rejected_streamlines = [], [], []
    show_curr_actor = True
    show_manager.start()

    # Early exit means everything else is rejected
    output_accepted_streamlines, output_rejected_streamlines = [], []
    missing = len(args.in_bundles) - len(choices) - len(sft_rejected_on_size)
    len_accepted = len(sft_accepted_on_size)
    rejected_streamlines.extend(range(len_accepted - missing,
                                      len_accepted))
    if missing > 0:
        logging.info('%s clusters automatically rejected from early exit',
                     missing)

    # Save accepted clusters (by GUI)
    for idx in accepted_streamlines:
        streamlines = sft_accepted_on_size[idx].streamlines
        output_accepted_streamlines.extend(streamlines)

        if args.out_accepted_dir:
            tmp_sft = StatefulTractogram(streamlines,
                                         sft_accepted_on_size[0],
                                         Space.RASMM)
            tmp_filename = os.path.join(args.out_accepted_dir,
                                        filename_accepted_on_size[idx])
            save_tractogram(tmp_sft, tmp_filename, bbox_valid_check=False)

    accepted_sft = StatefulTractogram(output_accepted_streamlines,
                                      sft_accepted_on_size[0],
                                      Space.RASMM)
    save_tractogram(accepted_sft, args.out_accepted, bbox_valid_check=False)

    # Save rejected clusters (by GUI)
    for idx in rejected_streamlines:
        streamlines = sft_accepted_on_size[idx].streamlines
        output_rejected_streamlines.extend(streamlines)

        if args.out_rejected_dir:
            tmp_sft = StatefulTractogram(streamlines,
                                         sft_accepted_on_size[0],
                                         Space.RASMM)
            tmp_filename = os.path.join(args.out_rejected_dir,
                                        filename_accepted_on_size[idx])
            save_tractogram(tmp_sft, tmp_filename, bbox_valid_check=False)

    # Save rejected clusters (by size)
    for idx in range(len(sft_rejected_on_size)):
        streamlines = sft_rejected_on_size[idx].streamlines
        output_rejected_streamlines.extend(streamlines)

        if args.out_rejected_dir:
            tmp_sft = StatefulTractogram(streamlines,
                                         sft_accepted_on_size[0],
                                         Space.RASMM)
            tmp_filename = os.path.join(args.out_rejected_dir,
                                        filename_rejected_on_size[idx])
            save_tractogram(tmp_sft, tmp_filename, bbox_valid_check=False)

    rejected_sft = StatefulTractogram(output_rejected_streamlines,
                                      sft_accepted_on_size[0],
                                      Space.RASMM)
    save_tractogram(rejected_sft, args.out_rejected, bbox_valid_check=False)


if __name__ == "__main__":
    main()
