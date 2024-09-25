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
  - Volume-weighted: Volume of the bundle.
  - Similarity: mean density??
      Uses pre-computed density maps, which can be obtained with
      >> scil_connectivity_hdf5_average_density_map.py
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
  - Mean DPS: Mean values in the data_per_streamline of each streamline in the
      bundles.

??? The bundles should be averaged version in the same space. This will
compute the weighted-dice between each node and their homologuous average
version.

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
                             assert_headers_compatible)
from scilpy.connectivity.connectivity import d


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter, )
    p.add_argument('in_hdf5',
                   help='Input filename for the hdf5 container (.h5).')
    p.add_argument('in_labels',
                   help='Labels file name (nifti).\n'
                        'This generates a NxN connectivity matrix, where N \n'
                        'is the number of values in in_labels.')

    g = p.add_argument_group("Output matrices options")
    g.add_argument('--volume', metavar='OUT_FILE',
                   help='Output file for the volume weighted matrix (.npy).')
    g.add_argument('--streamline_count', metavar='OUT_FILE',
                   help='Output file for the streamline count weighted matrix '
                        '(.npy).')
    g.add_argument('--length', metavar='OUT_FILE',
                   help='Output file for the length weighted matrix (.npy).')
    g.add_argument('--similarity', nargs=2,
                   metavar=('IN_FOLDER', 'OUT_FILE'),
                   help='Input folder containing the averaged bundle density\n'
                        'maps (.nii.gz) and output file for the similarity '
                        'weighted matrix (.npy).\n'
                        'The density maps should be named using the same '
                        'labels as in the hdf5 (LABEL1_LABEL2.nii.gz).')
    g.add_argument('--metrics', nargs=2, action='append',
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
                   help='Use density-weighting for the metric weighted'
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


def menage_a_faire(parser, args):
    # Verifying now most of the outputs.
    # Other will be verified later when we know the label combination.
    # Summarizing all options chosen by user in measures_to_compute.
    measures_to_compute = []
    measures_output_filename = []
    if args.volume:
        measures_to_compute.append('volume')
        measures_output_filename.append(args.volume)
    if args.streamline_count:
        measures_to_compute.append('streamline_count')
        measures_output_filename.append(args.streamline_count)
    if args.length:
        measures_to_compute.append('length')
        measures_output_filename.append(args.length)
    if args.similarity:
        measures_to_compute.append('similarity')
        measures_output_filename.append(args.similarity[1])

    # Adding measures from pre-computed metrics.
    dict_metrics_out_name = {}
    if args.metrics is not None:
        for in_name, out_name in args.metrics:
            # This is necessary to support more than one map for weighting
            measures_to_compute.append((in_name, nib.load(in_name)))
            dict_metrics_out_name[in_name] = out_name
            measures_output_filename.append(out_name)

    # Adding measures from lesions.
    dict_lesion_out_name = {}
    if args.lesion_load is not None:
        in_name = args.lesion_load[0]
        lesion_img = nib.load(in_name)
        lesion_data = get_data_as_mask(lesion_img, dtype=bool)
        lesion_atlas, _ = ndi.label(lesion_data)
        measures_to_compute.append(((in_name, np.unique(lesion_atlas)[1:]),
                                    nib.Nifti1Image(lesion_atlas,
                                                    lesion_img.affine)))

        out_name_1 = os.path.join(args.lesion_load[1], 'lesion_vol.npy')
        out_name_2 = os.path.join(args.lesion_load[1], 'lesion_count.npy')
        out_name_3 = os.path.join(args.lesion_load[1], 'lesion_sc.npy')

        dict_lesion_out_name[in_name + 'vol'] = out_name_1
        dict_lesion_out_name[in_name + 'count'] = out_name_2
        dict_lesion_out_name[in_name + 'sc'] = out_name_3
        measures_output_filename.extend([out_name_1, out_name_2, out_name_3])

    # Verifying all outputs that will be used for all measures.
    if not measures_to_compute:
        parser.error('No connectivity measures were selected, nothing '
                     'to compute.')

    logging.info('The following measures will be computed and save: {}'.format(
        measures_output_filename))

    if args.include_dps:
        if not os.path.isdir(args.include_dps):
            os.makedirs(args.include_dps)
        logging.info('data_per_streamline weighting is activated.')

    return measures_to_compute, dict_metrics_out_name, dict_lesion_out_name


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    coloredlogs.install(level=logging.getLevelName(args.verbose))

    # Verifications
    optional_volumes = []
    if args.metrics is not None:
        optional_volumes.extend([m[0] for m in args.metrics])
    if args.lesion_load is not None:
        optional_volumes.extend([ll[0] for ll in args.lesion_load])
    assert_inputs_exist(parser, [args.in_hdf5, args.in_labels],
                        [args.force_labels_list] + optional_volumes)
    if args.similarity is not None:
        # Note. Inputs in the --similarity folder are not checked yet!
        # But at least checking that the folder exists
        assert_inputs_dirs_exist(parser, [], args.similarity[0])
    assert_headers_compatible(parser, args.in_labels, optional_volumes)
    with h5py.File(args.in_hdf5, 'r') as hdf5:
        vol = nib.load(args.in_labels)
        assert_header_compatible_hdf5(hdf5, vol)

    (measures_to_compute, dict_metrics_out_name,
     dict_lesion_out_name) = menage_a_faire(parser, args)
    # assert_outputs_exist(parser, args, measures_to_compute)

    # Loading the data
    img_labels = nib.load(args.in_labels)
    data_labels = get_data_as_labels(img_labels)
    if not args.force_labels_list:
        labels_list = np.unique(data_labels)[1:].tolist()
    else:
        labels_list = np.loadtxt(
            args.force_labels_list, dtype=np.int16).tolist()
    logging.info("Found {} labels.".format(len(labels_list)))

    # Finding all connectivity combo (start-finish)
    comb_list = list(itertools.combinations(labels_list, r=2))
    if not args.no_self_connection:
        comb_list.extend(zip(labels_list, labels_list))

    # Running everything!
    nbr_cpu = validate_nbr_processes(parser, args)
    measures_dict_list = []
    compute_volume = args.volume is not None
    compute_length = args.length is not None
    if nbr_cpu == 1:
        for comb in comb_list:
            measures_dict_list.append(compute_connectivity_matrices_from_hdf5(
                args.in_hdf5, img_labels, comb[0], comb[1],
                compute_volume, compute_length, args.similarity, args.density_weighting,
                args.include_dps, args.min_lesion_vol))
    else:
        def set_num(counter):
            d.id = next(counter) + 1

        logging.info("PREPARING MULTIPOOLING: {}".format(comb_list))
        pool = multiprocessing.Pool(nbr_cpu, initializer=set_num, initargs=(itertools.count(),))

        # Dividing the process bundle by bundle
        measures_dict_list = pool.map(
            multi_proc_compute_connectivity_matrices_from_hdf5,
            zip(itertools.repeat(args.in_hdf5),
                itertools.repeat(img_labels),
                comb_list,
                itertools.repeat(compute_volume),
                itertools.repeat(compute_length),
                itertools.repeat(args.similarity),
                itertools.repeat(args.density_weighting),
                itertools.repeat(args.include_dps),
                itertools.repeat(args.min_lesion_vol)
                ))
        pool.close()
        pool.join()

    # Removing None entries (combinaisons that do not exist)
    # Fusing the multiprocessing output into a single dictionary
    measures_dict_list = [it for it in measures_dict_list if it is not None]
    if not measures_dict_list:
        raise ValueError('Empty matrix, no entries to save.')
    measures_dict = measures_dict_list[0]
    for dix in measures_dict_list[1:]:
        measures_dict.update(dix)

    if args.no_self_connection:
        total_elem = len(labels_list) ** 2 - len(labels_list)
        results_elem = len(measures_dict.keys()) * 2 - len(labels_list)
    else:
        total_elem = len(labels_list) ** 2
        results_elem = len(measures_dict.keys()) * 2

    logging.info('Out of {} possible nodes, {} contain value'.format(
        total_elem, results_elem))

    # Filling out all the matrices (symmetric) in the order of labels_list
    nbr_of_measures = len(list(measures_dict.values())[0])
    matrix = np.zeros((len(labels_list), len(labels_list), nbr_of_measures))

    for in_label, out_label in measures_dict:
        curr_node_dict = measures_dict[(in_label, out_label)]
        measures_ordering = list(curr_node_dict.keys())

        for i, measure in enumerate(curr_node_dict):
            in_pos = labels_list.index(in_label)
            out_pos = labels_list.index(out_label)
            matrix[in_pos, out_pos, i] = curr_node_dict[measure]
            matrix[out_pos, in_pos, i] = curr_node_dict[measure]

    # Saving the matrices separatly with the specified name or dps
    for i, measure in enumerate(measures_ordering):
        if measure == 'volume':
            matrix_basename = args.volume
        elif measure == 'streamline_count':
            matrix_basename = args.streamline_count
        elif measure == 'length':
            matrix_basename = args.length
        elif measure == 'similarity':
            matrix_basename = args.similarity[1]
        elif measure in dict_metrics_out_name:
            matrix_basename = dict_metrics_out_name[measure]
        elif measure in dict_lesion_out_name:
            matrix_basename = dict_lesion_out_name[measure]
        else:
            matrix_basename = measure

        np.save(matrix_basename, matrix[:, :, i])


if __name__ == "__main__":
    main()
