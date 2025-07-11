#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script computes a variety of measures in the form of connectivity
matrices. This script only generates matrices in the form of array, it does not
visualize or reorder the labels (node).

See also
>> scil_connectivity_compute_simple_matrix.py
which simply computes the connectivity matrix (either binary or with the
streamline count), directly from the endpoints.

In comparison, the current script A) uses more complex segmentation, and
B) outputs more matrices, using various metrics,

A) Connections segmentations
----------------------------
Segmenting a tractogram based on its endpoints is not as straighforward as one
could imagine. The endpoints could be outside any labelled region. This script
is made to follow
>> scil_tractogram_segment_connections_from_labels.py,
which already carefully segmented the connections.

The current script uses 1) the same labels list as input, 2) the resulting
pre-segmented tractogram in the hdf5 format, and 3) a text file containing the
list of labels that should be part of the matrices. The ordering of labels in
the matrices will follow the same order as the list.

B) Outputs
----------
Each connection can be seen as a 'bundle'.

  - Streamline count.
  - Length: mean streamline length (mm).
      Note that this matrix, as well as the volume-weighted, can be used to
      normalize a streamline count matrix in scil_connectivity_normalize.
  - Volume-weighted: Volume of the bundle.
  - Similarity: mean density.
      Uses pre-computed density maps, which can be obtained with
      >> scil_connectivity_hdf5_average_density_map.py
      The bundles should be averaged version in the same space. This will
      compute the weighted-dice between each node and their homologuous average
      version.
  - Any metric: You can provide your own maps through --metrics. The average
      non-zero value in the volume occupied by the bundle will be reported in
      the matrices nodes.
      Ex: --metrics FA.niigz fa.npy --metrics T1.nii.gz t1.npy
  - Lesions-related metrics: The option --lesion_load will compute 3
      lesion(s)-related matrices (saved in the chosen output directory):
      lesion_count.npy, lesion_vol.npy, and lesion_sc.npy. They represent the
      number of lesion, the total volume of lesion(s) and the total number of
      streamlines going through the lesion(s) for each bundle. See also:
      >> scil_analyse_lesion_load.py
      >> scil_lesions_info.py
  - Mean DPS: Mean values in the data_per_streamline of each streamline in the
      bundles.

What next?
==========
See our other scripts to help you achieve your goals:
  - Normalize a streamline-count matrix based on other matrices using
    scil_connectivity_normalize.
  - Compute a t-test between two groups of subjects using
    scil_connectivity_compare_populations.
  - See all our scripts starting with scil_connectivity_ for more ideas!

Formerly: scil_compute_connectivity.py
"""

import argparse
import itertools
import logging
import multiprocessing
import os

import coloredlogs
import h5py
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

from scilpy.connectivity.connectivity import \
    compute_connectivity_matrices_from_hdf5, \
    multi_proc_compute_connectivity_matrices_from_hdf5
from scilpy.image.labels import get_data_as_labels
from scilpy.io.hdf5 import assert_header_compatible_hdf5
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             validate_nbr_processes, assert_inputs_dirs_exist,
                             assert_headers_compatible,
                             assert_output_dirs_exist_and_empty)
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_hdf5',
                   help='Input filename for the hdf5 container (.h5).')
    p.add_argument('in_labels',
                   help='Labels file name (nifti).\n'
                        'This generates a NxN connectivity matrix, where N \n'
                        'is the number of values in in_labels.')

    g = p.add_argument_group("Output matrices options")
    g.add_argument('--volume', metavar='OUT_FILE',
                   help='Output file for the volume weighted matrix (.npy), '
                        'computed in mm3.')
    g.add_argument('--streamline_count', metavar='OUT_FILE',
                   help='Output file for the streamline count weighted matrix '
                        '(.npy).')
    g.add_argument('--length', metavar='OUT_FILE',
                   help='Output file for the length weighted matrix (.npy), '
                        'weighted in mm.')
    g.add_argument('--similarity', nargs=2,
                   metavar=('IN_FOLDER', 'OUT_FILE'),
                   help='Input folder containing the averaged bundle density\n'
                        'maps (.nii.gz) and output file for the similarity '
                        'weighted matrix (.npy).\n'
                        'The density maps should be named using the same '
                        'labels as in the hdf5 (LABEL1_LABEL2.nii.gz).')
    g.add_argument('--metrics', nargs=2, action='append', default=[],
                   metavar=('IN_FILE', 'OUT_FILE'),
                   help='Input (.nii.gz). and output file (.npy) for a metric '
                        'weighted matrix.')
    g.add_argument('--lesion_load', nargs=2, metavar=('IN_FILE', 'OUT_DIR'),
                   help='Input binary mask (.nii.gz) and output directory '
                        'for all lesion-related matrices.')
    g.add_argument('--include_dps', metavar='OUT_DIR',
                   help='Save matrices from data_per_streamline in the output '
                        'directory.\nCOMMIT-related values will be summed '
                        'instead of averaged.\nWill always overwrite files.')

    g = p.add_argument_group("Processing options")
    g.add_argument('--min_lesion_vol', type=float, default=7,
                   help='Minimum lesion volume in mm3 [%(default)s].')
    g.add_argument('--density_weighting', action="store_true",
                   help='Use density-weighting for the metric weighted '
                        'matrix.')
    g.add_argument('--no_self_connection', action="store_true",
                   help='Eliminate the diagonal from the matrices.')
    g.add_argument('--force_labels_list',
                   help='Path to a labels list (.txt) in case of missing '
                        'labels in the atlas.')

    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def check_inputs_outputs(parser, args):
    optional_input_volumes = []
    optional_output_matrices = [args.volume, args.streamline_count,
                                args.length]
    out_dirs = [args.include_dps]
    if args.metrics is not None:
        optional_input_volumes.extend([m[0] for m in args.metrics])
        optional_output_matrices.extend([m[1] for m in args.metrics])
    if args.lesion_load is not None:
        optional_input_volumes.append(args.lesion_load[0])
        optional_output_matrices.append(args.lesion_load[1])
    if args.similarity is not None:
        optional_output_matrices.append(args.similarity[1])
        # Note. Inputs in the --similarity folder are not checked yet!
        # But at least checking that the folder exists
        assert_inputs_dirs_exist(parser, [], args.similarity[0])
    if args.lesion_load is not None:
        out_dirs.append(args.lesion_load[1])

    # Inputs
    assert_inputs_exist(parser, [args.in_hdf5, args.in_labels],
                        [args.force_labels_list] + optional_input_volumes)

    # Headers
    assert_headers_compatible(parser, args.in_labels, optional_input_volumes)
    with h5py.File(args.in_hdf5, 'r') as hdf5:
        vol = nib.load(args.in_labels)
        assert_header_compatible_hdf5(hdf5, vol)

    # Outputs
    assert_outputs_exist(parser, args, [], optional_output_matrices)
    assert_output_dirs_exist_and_empty(parser, args, [], out_dirs)
    for m in optional_output_matrices:
        if m is not None and m[-4:] != '.npy':
            parser.error("Expecting .npy for the output matrix, got: {}"
                         .format(m))


def fill_matrix_and_save(measures_dict, labels_list, measure_keys, filenames):
    matrix = np.zeros((len(labels_list), len(labels_list), len(measure_keys)))

    # Run one loop on node. Fill all matrices at once.
    for label_key, node_values in measures_dict.items():
        in_label, out_label = label_key
        for i, measure_key in enumerate(measure_keys):
            in_pos = labels_list.index(in_label)
            out_pos = labels_list.index(out_label)
            matrix[in_pos, out_pos, i] = node_values[measure_key]
            matrix[out_pos, in_pos, i] = node_values[measure_key]

    for i, f in enumerate(filenames):
        logging.info("Saving resulting {} in file {}"
                     .format(measure_keys[i], f))
        np.save(f, matrix[:, :, i])


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    coloredlogs.install(level=logging.getLevelName(args.verbose))

    # Verifications
    check_inputs_outputs(parser, args)

    # Verifying that at least one option is selected
    compute_volume = args.volume is not None
    compute_streamline_count = args.streamline_count is not None
    compute_length = args.length is not None
    similarity_directory = args.similarity[0] if args.similarity else None
    if not (compute_volume or compute_streamline_count or compute_length or
            similarity_directory is not None or len(args.metrics) > 0 or
            args.lesion_load is not None or args.include_dps):
        parser.error("Please select at least one output matrix to compute.")

    # Loading the data
    img_labels = nib.load(args.in_labels)
    data_labels = get_data_as_labels(img_labels)
    if not args.force_labels_list:
        labels_list = np.unique(data_labels)[1:].tolist()
    else:
        labels_list = np.loadtxt(
            args.force_labels_list, dtype=np.int16).tolist()
    logging.info("Found {} labels.".format(len(labels_list)))

    # Not preloading the similarity (density) files, as there are many
    # (one per node). Can be loaded and discarded when treating each node.

    # Preloading the metrics here (FA, T1) to avoid reloading for each
    # node! But if there are many metrics, this could be heavy to keep in
    # memory, especially if multiprocessing is used. Still probably better.
    metrics_data = []
    metrics_names = []
    for m in args.metrics:
        metrics_names.append(m[1])
        metrics_data.append(nib.load(m[0]).get_fdata(dtype=np.float64))

    # Preloading the lesion file
    lesion_data = None
    if args.lesion_load is not None:
        lesion_img = nib.load(args.lesion_load[0])
        lesion_data = get_data_as_mask(lesion_img, dtype=bool)
        lesion_atlas, _ = ndi.label(lesion_data)
        lesion_labels = np.unique(lesion_atlas)[1:]
        atlas_img = nib.Nifti1Image(lesion_atlas, lesion_img.affine)
        lesion_data = (lesion_labels, atlas_img)

    # Finding all connectivity combo (start-finish)
    comb_list = list(itertools.combinations(labels_list, r=2))
    if not args.no_self_connection:
        comb_list.extend(zip(labels_list, labels_list))

    # Running everything!
    nbr_cpu = validate_nbr_processes(parser, args)
    outputs = []
    if nbr_cpu == 1:
        for comb in comb_list:
            outputs.append(compute_connectivity_matrices_from_hdf5(
                args.in_hdf5, img_labels, comb[0], comb[1],
                compute_volume, compute_streamline_count, compute_length,
                similarity_directory, metrics_data, metrics_names,
                lesion_data, args.include_dps, args.density_weighting,
                args.min_lesion_vol))
    else:
        pool = multiprocessing.Pool(nbr_cpu)

        # Dividing the process bundle by bundle
        outputs = pool.map(
            multi_proc_compute_connectivity_matrices_from_hdf5,
            zip(itertools.repeat(args.in_hdf5),
                itertools.repeat(img_labels),
                comb_list,
                itertools.repeat(compute_volume),
                itertools.repeat(compute_streamline_count),
                itertools.repeat(compute_length),
                itertools.repeat(similarity_directory),
                itertools.repeat(metrics_data),
                itertools.repeat(metrics_names),
                itertools.repeat(lesion_data),
                itertools.repeat(args.include_dps),
                itertools.repeat(args.density_weighting),
                itertools.repeat(args.min_lesion_vol)
                ))
        pool.close()
        pool.join()

    # Removing None entries (combinaisons that do not exist)
    outputs = [it for it in outputs if it is not None]
    if len(outputs) == 0:
        raise ValueError('No connection found at all! Matrices would be '
                         'all-zeros. Exiting.')

    measures_dict_list = [it[0] for it in outputs]
    dps_keys = [it[1] for it in outputs]

    # Verify that all bundles had the same dps_keys
    if len(dps_keys) > 1 and not dps_keys[1:] == dps_keys[:-1]:
        raise ValueError("DPS keys not consistant throughout the hdf5 "
                         "connections. Verify your tractograms, or do not "
                         "use --include_dps.")
    dps_keys = dps_keys[0]

    # Fusing the multiprocessing output into a single dictionary
    measures_dict = {}
    for node in measures_dict_list:
        measures_dict.update(node)

    # Filling out all the matrices (symmetric) in the order of labels_list
    keys = []
    filenames = []
    if compute_volume:
        keys.append('volume_mm3')
        filenames.append(args.volume)
    if compute_length:
        keys.append('length_mm')
        filenames.append(args.length)
    if compute_streamline_count:
        keys.append('streamline_count')
        filenames.append(args.streamline_count)
    if similarity_directory is not None:
        keys.append('similarity')
        filenames.append(args.similarity[1])
    if len(args.metrics) > 0:
        keys.extend(metrics_names)
        filenames.extend([m[1] for m in args.metrics])
    if args.lesion_load is not None:
        keys.extend(['lesion_vol', 'lesion_count', 'lesion_streamline_count'])
        filenames.extend(
            [os.path.join(args.lesion_load[1], 'lesion_vol.npy'),
             os.path.join(args.lesion_load[1], 'lesion_count.npy'),
             os.path.join(args.lesion_load[1], 'lesion_sc.npy')])
    if args.include_dps:
        keys.extend(dps_keys)
        filenames.extend([os.path.join(args.include_dps, "{}.npy".format(k))
                          for k in dps_keys])
    fill_matrix_and_save(measures_dict, labels_list, keys, filenames)


if __name__ == "__main__":
    main()
