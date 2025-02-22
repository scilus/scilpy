#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Divide a tractogram into its various connections using a brain parcellation
(labels). Allows using our connectivity scripts. See for instance:
>>> scil_connectivity_compute_matrices.py

The hdf5 output format allows to store other information required for
connectivity, such as the associated labels. To visualize the segmented
bundles, it is possible to convert the result using:
>>> scil_tractogram_convert_hdf5_to_trk.py

Cleaning your tractogram
------------------------
Outliers may influence strongly the connectivity analysis. We recommand
cleaning your tractogram as much as possible beforehand. Options are offered in
this script and are activated by default.
For the --outlier_threshold option, the default is our recommended trade-off
for a good freesurfer parcellation. With smaller parcels (brainnetome, glasser)
the threshold should most likely be reduced.

See also:
    - scil_tractogram_filter_by_anatomy.py
    - scil_tractogram_filter_by_length.py
    - scil_tractogram_filter_by_roi.py
    - scil_tractogram_detect_loops.py

The segmentation process
------------------------
Segmenting a tractogram based on its endpoints is not as straighforward as one
could imagine. The endpoints could be outside any labelled region.

The current strategy is to keep the longest streamline segment connecting 2
regions. If the streamline crosses other gray matter regions before reaching
its final connected region, the kept connection is still the longest. This is
robust to compressed streamlines.

NOTE: this script can take a while to run. Please be patient.
Example: on a tractogram with 1.8M streamlines, running on a SSD:
- 15 minutes without post-processing, only saving final bundles.
- 30 minutes with full post-processing, only saving final bundles.
- 60 minutes with full post-processing, saving all possible files.

Verifying the results
---------------------
Good candidate connections to use for quality control (QC) are 1) the brainstem
to precentral gyrus connection and 2) the precentral left to precentral right
connection, or equivalent in your parcellation.

Note that the final streamlines saved in the hdf5 are cut between the two
associated labels (points after the ROIs are removed, if any).

The output hdf5 architecture (nerdy stuff)
----------------------------
The output file is a hdf5 (.h5) where each bundle is a group with key
'LABEL1_LABEL2' and each. The array_sequence format cannot be stored directly
in a hdf5, so each group is composed of 'data', 'offsets' and 'lengths' from
the array_sequence. The 'data' is stored in VOX/CORNER for simplicity and
efficiency.

Formerly: scil_decompose_connectivity.py
"""
import argparse
import logging
import os
import time

import coloredlogs
import h5py
import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.io.hdf5 import construct_hdf5_header
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_bbox_arg, add_overwrite_arg,
                             add_processes_arg, add_verbose_arg,
                             add_reference_arg, assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty,
                             validate_nbr_processes, assert_headers_compatible)
from scilpy.tractanalysis.connectivity_segmentation import (
    compute_connectivity,
    construct_hdf5_from_connectivity,
    extract_longest_segments_from_profile)
from scilpy.tractograms.uncompress import streamlines_to_voxel_coordinates
from scilpy.version import version_string


def _get_output_paths(args):
    paths = {'raw': os.path.join(args.out_dir, 'raw_connections'),
             'final': os.path.join(args.out_dir, 'final_connections'),
             'invalid_length': os.path.join(args.out_dir, 'invalid_length'),
             'valid_length': os.path.join(args.out_dir, 'valid_length'),
             'loops': os.path.join(args.out_dir, 'loops'),
             'outliers': os.path.join(args.out_dir, 'outliers'),
             'qb_curv': os.path.join(args.out_dir, 'qb_curv'),
             'no_loops': os.path.join(args.out_dir, 'no_loops'),
             'inliers': os.path.join(args.out_dir, 'inliers')}

    return paths


def _get_saving_options(args):
    saving_options = {'raw': args.save_raw_connections,
                      'intermediate': args.save_intermediate,
                      'discarded': args.save_discarded,
                      'final': args.save_final}

    return saving_options


def _create_required_output_dirs(args, out_paths):
    if not args.out_dir:
        return

    os.mkdir(out_paths['final'])

    if args.save_raw_connections:
        os.mkdir(out_paths['raw'])

    if args.save_discarded:
        os.mkdir(out_paths['loops'])
        os.mkdir(out_paths['outliers'])
        os.mkdir(out_paths['qb_curv'])
        os.mkdir(out_paths['invalid_length'])

    if args.save_intermediate:
        os.mkdir(out_paths['no_loops'])
        os.mkdir(out_paths['inliers'])
        os.mkdir(out_paths['valid_length'])


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)
    p.add_argument('in_tractograms', nargs='+',
                   help='Tractogram filename (s). Format must be one of \n'
                        'trk, tck, vtk, fib, dpy.\n'
                        'If you have many tractograms for a single subject '
                        '(ex, coming \nfrom Ensemble Tracking), we will '
                        'merge them together.')
    p.add_argument('in_labels',
                   help='Labels file name (nifti). Labels must have 0 as '
                        'background. Volumes must have isotropic voxels.')
    p.add_argument('out_hdf5',
                   help='Output hdf5 file (.h5).')

    post_proc = p.add_argument_group('Post-processing options')
    post_proc.add_argument('--no_pruning', action='store_true',
                           help='If set, will NOT prune on length.\n'
                                'Length criteria in --min_length, '
                                '--max_length.')
    post_proc.add_argument('--no_remove_loops', action='store_true',
                           help='If set, will NOT remove streamlines making '
                                'loops.\nAngle criteria based on '
                                '--loop_max_angle.')
    post_proc.add_argument('--no_remove_outliers', action='store_true',
                           help='If set, will NOT remove outliers using QB.\n'
                                'Criteria based on --outlier_threshold.')
    post_proc.add_argument('--no_remove_curv_dev', action='store_true',
                           help='If set, will NOT remove streamlines that '
                                'deviate from the mean curvature.\n'
                                'Threshold based on --curv_qb_distance.')

    pr = p.add_argument_group('Pruning options')
    pr.add_argument('--min_length', type=float, default=20.,
                    help='Pruning minimal segment length. [%(default)s]')
    pr.add_argument('--max_length', type=float, default=200.,
                    help='Pruning maximal segment length. [%(default)s]')

    og = p.add_argument_group('Outliers and loops options')
    og.add_argument('--outlier_threshold', type=float, default=0.6,
                    help='Outlier removal threshold when using hierarchical '
                         'QB. [%(default)s]')
    og.add_argument('--loop_max_angle', type=float, default=330.,
                    help='Maximal winding angle over which a streamline is '
                         'considered as looping. [%(default)s]')
    og.add_argument('--curv_qb_distance', type=float, default=10.,
                    help='Clustering threshold for centroids curvature '
                         'filtering with QB. [%(default)s]')

    s = p.add_argument_group('Saving options')
    s.add_argument('--out_dir',
                   help='Output directory for each file based on options '
                        'below, as separate files (.trk).')
    s.add_argument('--save_raw_connections', action='store_true',
                   help='If set, will save all raw cut connections in a '
                        'subdirectory.')
    s.add_argument('--save_intermediate', action='store_true',
                   help='If set, will save the intermediate results of '
                        'filtering.')
    s.add_argument('--save_discarded', action='store_true',
                   help='If set, will save discarded streamlines in '
                        'subdirectories.\n'
                        'Includes loops, outliers and qb_loops.')
    s.add_argument('--save_final', action='store_true',
                   help='If set, will save the final bundles (connections) '
                        'on disk (.trk) as well as in the hdf5.\n'
                        'If this is not set, you can also get the final '
                        'bundles later, using:\n'
                        'scil_tractogram_convert_hdf5_to_trk.py.')

    p.add_argument('--out_labels_list', metavar='OUT_FILE',
                   help='Save the labels list as text file.\n'
                        'Needed for scil_connectivity_compute_matrices.py and '
                        'others.')

    add_reference_arg(p)
    add_bbox_arg(p)
    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    coloredlogs.install(level=logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, args.in_tractograms + [args.in_labels],
                        args.reference)
    assert_outputs_exist(parser, args, args.out_hdf5, args.out_labels_list)
    assert_headers_compatible(parser, args.in_tractograms + [args.in_labels],
                              [], args.reference)
    nbr_cpu = validate_nbr_processes(parser, args)

    # HDF5 will not overwrite the file
    if os.path.isfile(args.out_hdf5):
        os.remove(args.out_hdf5)

    out_paths = {}
    if (args.save_raw_connections or args.save_intermediate
            or args.save_discarded or args.save_final):
        if not args.out_dir:
            parser.error('To save outputs in the streamlines form, provide '
                         'the output directory using --out_dir.')
        out_paths = _get_output_paths(args)
    elif args.out_dir:
        logging.info("--out_dir will not be created, as there is nothing to "
                     "be saved.")
        args.out_dir = None

    if args.out_dir:
        if os.path.abspath(args.out_dir) == os.getcwd():
            parser.error('Do not use the current path as output directory.')
        assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                           create_dir=True)
        _create_required_output_dirs(args, out_paths)

    # Load everything
    img_labels = nib.load(args.in_labels)
    data_labels = get_data_as_labels(img_labels)
    real_labels = np.unique(data_labels)[1:]   # Removing the background 0.
    if args.out_labels_list:
        np.savetxt(args.out_labels_list, real_labels, fmt='%i')

    # Voxel size must be isotropic, for speed/performance considerations
    vox_sizes = img_labels.header.get_zooms()
    if not np.allclose(np.mean(vox_sizes), vox_sizes, atol=1e-01):
        parser.error('Labels must be isotropic')

    logging.info('*** Loading streamlines ***')
    time1 = time.time()
    sft = None
    for in_tractogram in args.in_tractograms:
        if sft is None:
            sft = load_tractogram_with_reference(parser, args, in_tractogram)
        else:
            sft += load_tractogram_with_reference(parser, args, in_tractogram)

    time2 = time.time()
    logging.info('    Loading {} streamlines took {} sec.'.format(
        len(sft), round(time2 - time1, 2)))

    sft.to_vox()
    sft.to_corner()

    # Main processing.
    # Get the indices of the voxels traversed by each streamline
    logging.info('*** Computing voxels traversed by each streamline ***')
    time1 = time.time()
    indices, points_to_idx = streamlines_to_voxel_coordinates(
        sft.streamlines,
        return_mapping=True
    )
    time2 = time.time()
    logging.info('    Streamlines intersection took {} sec.'.format(
        round(time2 - time1, 2)))

    # Compute the connectivity mapping
    logging.info('*** Computing connectivity information ***')
    time1 = time.time()
    con_info = compute_connectivity(indices, data_labels, real_labels,
                                    extract_longest_segments_from_profile)
    time2 = time.time()
    logging.info('    Connectivity computation took {} sec.'.format(
        round(time2 - time1, 2)))

    logging.info('*** Starting connection post-processing and saving. ***')
    logging.info('    This can be long, be patient.')
    time1 = time.time()
    with h5py.File(args.out_hdf5, 'w') as hdf5_file:
        construct_hdf5_header(hdf5_file, sft)
        prune_length = not args.no_pruning
        remove_loops = not args.no_remove_loops
        remove_outliers = not args.no_remove_outliers
        remove_curv_dev = not args.no_remove_curv_dev
        construct_hdf5_from_connectivity(
            sft, indices, points_to_idx, real_labels, con_info,
            hdf5_file, _get_saving_options(args), out_paths,
            prune_length, args.min_length, args.max_length,
            remove_loops, args.loop_max_angle,
            remove_outliers, args.outlier_threshold,
            remove_curv_dev, args.curv_qb_distance,
            nbr_cpu)
    time2 = time.time()
    logging.info(
        '    Connections post-processing and saving took {} sec.'.format(
            round(time2 - time1, 2)))


if __name__ == "__main__":
    main()
