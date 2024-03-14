#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Render clusters sequentially to either accept or reject them based on
    visual inspection. Useful for cleaning bundles for RBx, BST or for figures.
    The VTK window does not handle well opacity of streamlines, this is a
    normal rendering behavior.
    Often use in pair with scil_tractogram_qbx.py.

    Key mapping:
    - a/A: accept displayed clusters
    - r/R: reject displayed clusters
    - z/Z: Rewing one element
    - c/C: Stop rendering of the background concatenation of streamlines
    - q/Q: Early window exist, everything remaining will be rejected
"""


import argparse
import os
import logging

from fury import window, actor, interactor
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_bbox_arg,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty,
                             assert_headers_compatible)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
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
                   help='Linewidth of the background streamlines [%(default)s]'
                   '.')
    p.add_argument('--clusters_linewidth', type=float, default=1,
                   help='Linewidth of the current cluster [%(default)s].')

    add_reference_arg(p)
    add_bbox_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    # Callback required for FURY
    def keypress_callback(obj, _):
        key = obj.GetKeySym().lower()
        nonlocal clusters_linewidth, background_linewidth
        nonlocal curr_streamlines_actor, concat_streamlines_actor, \
            show_curr_actor
        iterator = len(accepted_streamlines) + len(rejected_streamlines)
        iren = interactor_style.GetInteractor()
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
            iren.TerminateApp()
            del renwin, iren
            if iterator < len(sft_accepted_on_size):
                logging.warning(
                    'Early exit, everything remaining to be rejected.')
            return

        if key in ['a', 'r'] and iterator < len(sft_accepted_on_size):
            if key == 'a':
                accepted_streamlines.append(iterator)
                choices.append('a')
                logging.info('Accepted file {}'.format(
                             filename_accepted_on_size[iterator]))
            elif key == 'r':
                rejected_streamlines.append(iterator)
                choices.append('r')
                logging.info('Rejected file {}'.format(
                             filename_accepted_on_size[iterator]))
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

    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, args.in_bundles, args.reference)
    assert_outputs_exist(parser, args, [args.out_accepted, args.out_rejected])
    assert_headers_compatible(parser, args.in_bundles,
                              reference=args.reference)

    if args.out_accepted_dir:
        assert_output_dirs_exist_and_empty(parser, args,
                                           args.out_accepted_dir,
                                           create_dir=True)
    if args.out_rejected_dir:
        assert_output_dirs_exist_and_empty(parser, args,
                                           args.out_rejected_dir,
                                           create_dir=True)

    if args.min_cluster_size < 1:
        parser.error('Minimum cluster size must be at least 1.')

    clusters_linewidth = args.clusters_linewidth
    background_linewidth = args.background_linewidth

    # To accelerate procedure, clusters can be discarded based on size
    # Concatenation is to give spatial context
    sft_accepted_on_size, filename_accepted_on_size = [], []
    sft_rejected_on_size, filename_rejected_on_size = [], []
    concat_streamlines = []

    ref_bundle = load_tractogram_with_reference(
        parser, args, args.in_bundles[0])

    for filename in args.in_bundles:
        basename = os.path.basename(filename)
        sft = load_tractogram_with_reference(parser, args, filename)
        if len(sft) >= args.min_cluster_size:
            sft_accepted_on_size.append(sft)
            filename_accepted_on_size.append(basename)
            concat_streamlines.extend(sft.streamlines)
        else:
            logging.info('File {} has {} streamlines,'
                         'automatically rejected.'.format(filename, len(sft)))
            sft_rejected_on_size.append(sft)
            filename_rejected_on_size.append(basename)

    if not filename_accepted_on_size:
        parser.error('No cluster survived the cluster_size threshold.')

    logging.info('{} clusters to be classified.'.format(
                 len(sft_accepted_on_size)))
    # The clusters are sorted by size for simplicity/efficiency
    tuple_accepted = zip(*sorted(zip(sft_accepted_on_size,
                                     filename_accepted_on_size),
                                 key=lambda x: len(x[0]), reverse=True))
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
    missing = len(args.in_bundles) - len(choices) - len(sft_rejected_on_size)
    len_accepted = len(sft_accepted_on_size)
    rejected_streamlines.extend(range(len_accepted - missing,
                                      len_accepted))
    if missing > 0:
        logging.info('{} clusters automatically rejected'
                     'from early exit'.format(missing))

    # Save accepted clusters (by GUI)
    accepted_streamlines = save_clusters(sft_accepted_on_size,
                                         accepted_streamlines,
                                         args.out_accepted_dir,
                                         filename_accepted_on_size,
                                         args.bbox_check)

    accepted_sft = StatefulTractogram(accepted_streamlines,
                                      sft_accepted_on_size[0],
                                      Space.RASMM)
    save_tractogram(accepted_sft, args.out_accepted,
                    bbox_valid_check=args.bbox_check)

    # Save rejected clusters (by GUI)
    rejected_streamlines = save_clusters(sft_accepted_on_size,
                                         rejected_streamlines,
                                         args.out_rejected_dir,
                                         filename_accepted_on_size,
                                         args.bbox_check)

    # Save rejected clusters (by size)
    rejected_streamlines.extend(save_clusters(sft_rejected_on_size,
                                              range(len(sft_rejected_on_size)),
                                              args.out_rejected_dir,
                                              filename_rejected_on_size,
                                              args.bbox_check))

    rejected_sft = StatefulTractogram(rejected_streamlines,
                                      sft_accepted_on_size[0],
                                      Space.RASMM)
    save_tractogram(rejected_sft, args.out_rejected,
                    bbox_valid_check=args.bbox_check)


def save_clusters(cluster_lists, indexes_list, directory, basenames_list,
                  bbox_check):
    output_streamlines = []
    for idx in indexes_list:
        streamlines = cluster_lists[idx].streamlines
        output_streamlines.extend(streamlines)

        if directory:
            tmp_sft = StatefulTractogram(streamlines,
                                         cluster_lists[0],
                                         Space.RASMM)
            tmp_filename = os.path.join(directory,
                                        basenames_list[idx])
            save_tractogram(tmp_sft, tmp_filename,
                            bbox_valid_check=bbox_check)

    return output_streamlines


if __name__ == "__main__":
    main()
