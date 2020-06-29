#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script computes a variety of measures in the form of connectivity
matrices. This script is made to follow scil_decompose_connectivity and
uses the same labels list as input.

The script expects a folder containing all relevants bundles following the
naming convention LABEL1_LABEL2.trk and a text file containing the list of
labels that should be part of the matrices. The ordering of labels in the
matrices will follow the same order as the list.
This script only generates matrices in the form of array, does not visualize
or reorder the labels (node).

The parameter --similarity expects a folder with density maps (LABEL1_LABEL2.nii.gz)
following the same naming convention as the input directory.
The bundles should be averaged version in the same space. This will
compute the weighted-dice between each node and their homologuous average
version.

The parameters --metrics can be used more than once and expect a map (t1, fa,
etc.) in the same space and each will generate a matrix. The average value in
the volume occupied by the bundle will be reported in the matrices nodes.

The parameters --maps can be used more than once and expect a folder with
pre-computed maps (LABEL1_LABEL2.nii.gz) following the same naming convention
as the input directory. Each will generate a matrix. The average non-zeros
value in the map will be reported in the matrices nodes.
"""

import argparse
import copy
import itertools
import multiprocessing
import logging
import os

import coloredlogs
from dipy.io.utils import is_header_compatible, get_reference_info
from dipy.tracking.streamlinespeed import length
import h5py
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_label
from scilpy.io.streamlines import reconstruct_streamlines_from_hdf5
from scilpy.io.utils import (add_overwrite_arg, add_processes_arg,
                             add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             validate_nbr_processes)
from scilpy.tractanalysis.reproducibility_measures import compute_bundle_adjacency_voxel
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


def load_node_nifti(directory, in_label, out_label, ref_img):
    in_filename = os.path.join(directory,
                               '{}_{}.nii.gz'.format(in_label, out_label))

    if os.path.isfile(in_filename):
        if not is_header_compatible(in_filename, ref_img):
            raise IOError('{} do not have a compatible header'.format(
                in_filename))
        return nib.load(in_filename).get_fdata(dtype=np.float64)

    return None


def _processing_wrapper(args):
    hdf5_filename = args[0]
    labels_img = args[1]
    in_label, out_label = args[2]
    measures_to_compute = copy.copy(args[3])
    if args[4] is not None:
        similarity_directory = args[4][0]
    weighted = args[5]
    include_dps = args[6]

    hdf5_file = h5py.File(hdf5_filename, 'r')
    key = '{}_{}'.format(in_label, out_label)
    if key not in hdf5_file:
        return
    streamlines = reconstruct_streamlines_from_hdf5(hdf5_file, key)

    affine, dimensions, voxel_sizes, _ = get_reference_info(labels_img)
    measures_to_return = {}

    if not (np.allclose(hdf5_file.attrs['affine'], affine)
            and np.allclose(hdf5_file.attrs['dimensions'], dimensions)):
        raise ValueError('Provided hdf5 have incompatible headers.')

    # Precompute to save one transformation, insert later
    if 'length' in measures_to_compute:
        streamlines_copy = list(streamlines)
        # scil_decompose_connectivity.py requires isotropic voxels
        mean_length = np.average(length(streamlines_copy))*voxel_sizes[0]

    # If density is not required, do not compute it
    # Only required for volume, similarity and any metrics
    if not ((len(measures_to_compute) == 1 and
             ('length' in measures_to_compute or
              'streamline_count' in measures_to_compute)) or
            (len(measures_to_compute) == 2 and
             ('length' in measures_to_compute and
              'streamline_count' in measures_to_compute))):

        density = compute_tract_counts_map(streamlines,
                                           dimensions)

    if 'volume' in measures_to_compute:
        measures_to_return['volume'] = np.count_nonzero(density) * \
            np.prod(voxel_sizes)
        measures_to_compute.remove('volume')
    if 'streamline_count' in measures_to_compute:
        measures_to_return['streamline_count'] = len(streamlines)
        measures_to_compute.remove('streamline_count')
    if 'length' in measures_to_compute:
        measures_to_return['length'] = mean_length
        measures_to_compute.remove('length')
    if 'similarity' in measures_to_compute and similarity_directory:
        density_sim = load_node_nifti(similarity_directory,
                                      in_label, out_label,
                                      labels_img)
        if density_sim is None:
            ba_vox = 0
        else:
            ba_vox = compute_bundle_adjacency_voxel(density, density_sim)

        measures_to_return['similarity'] = ba_vox
        measures_to_compute.remove('similarity')

    for measure in measures_to_compute:
        if isinstance(measure, str) and os.path.isdir(measure):
            map_dirname = measure
            map_data = load_node_nifti(map_dirname,
                                       in_label, out_label,
                                       labels_img)
            measures_to_return[map_dirname] = np.average(
                map_data[map_data > 0])
        elif isinstance(measure, tuple) and os.path.isfile(measure[0]):
            metric_filename = measure[0]
            metric_img = measure[1]
            if not is_header_compatible(metric_img, labels_img):
                logging.error('{} do not have a compatible header'.format(
                    metric_filename))
                raise IOError

            metric_data = metric_img.get_fdata(dtype=np.float64)
            if weighted:
                density = density / np.max(density)
                voxels_value = metric_data * density
                voxels_value = voxels_value[voxels_value > 0]
            else:
                voxels_value = metric_data[density > 0]

            measures_to_return[metric_filename] = np.average(voxels_value)

    if include_dps:
        for dps_key in hdf5_file[key].keys():
            if dps_key not in ['data', 'offsets', 'lengths']:
                out_file = os.path.join(include_dps, dps_key)
                measures_to_return[out_file] = np.average(
                    hdf5_file[key][dps_key])

    return {(in_label, out_label): measures_to_return}


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,)
    p.add_argument('in_hdf5',
                   help='Input filename for the hdf5 container (.h5).\n'
                        'Obtained from scil_decompose_connectivity.py.')
    p.add_argument('in_labels',
                   help='Labels file name (nifti).\n'
                        'This generates a NxN connectivity matrix.')
    p.add_argument('--volume', metavar='OUT_FILE',
                   help='Output file for the volume weighted matrix (.npy).')
    p.add_argument('--streamline_count', metavar='OUT_FILE',
                   help='Output file for the streamline count weighted matrix '
                        '(.npy).')
    p.add_argument('--length', metavar='OUT_FILE',
                   help='Output file for the length weighted matrix (.npy).')
    p.add_argument('--similarity', nargs=2,
                   metavar=('IN_FOLDER', 'OUT_FILE'),
                   help='Input folder containing the averaged bundle density\n'
                        'maps (.nii.gz) and output file for the similarity '
                        'weighted matrix (.npy).')
    p.add_argument('--maps', nargs=2,  action='append',
                   metavar=('IN_FOLDER', 'OUT_FILE'),
                   help='Input folder containing pre-computed maps (.nii.gz)\n'
                        'and output file for the weighted matrix (.npy).')
    p.add_argument('--metrics', nargs=2, action='append',
                   metavar=('IN_FILE', 'OUT_FILE'),
                   help='Input (.nii.gz). and output file (.npy) for a metric '
                        'weighted matrix.')

    p.add_argument('--density_weighting', action="store_true",
                   help='Use density-weighting for the metric weighted matrix.')
    p.add_argument('--no_self_connection', action="store_true",
                   help='Eliminate the diagonal from the matrices.')
    p.add_argument('--include_dps', metavar='OUT_DIR',
                   help='Save matrices from data_per_streamline in the output '
                        'directory.\nWill always overwrite files.')
    p.add_argument('--force_labels_list',
                   help='Path to a labels list (.txt) in case of missing '
                        'labels in the atlas.')

    add_processes_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_hdf5, args.in_labels],
                        args.force_labels_list)

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)
    coloredlogs.install(level=log_level)

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

    dict_maps_out_name = {}
    if args.maps is not None:
        for in_folder, out_name in args.maps:
            measures_to_compute.append(in_folder)
            dict_maps_out_name[in_folder] = out_name
            measures_output_filename.append(out_name)

    dict_metrics_out_name = {}
    if args.metrics is not None:
        for in_name, out_name in args.metrics:
            # Verify that all metrics are compatible with each other
            if not is_header_compatible(args.metrics[0][0], in_name):
                raise IOError('Metrics {} and  {} do not share a compatible '
                              'header'.format(args.metrics[0][0], in_name))

            # This is necessary to support more than one map for weighting
            measures_to_compute.append((in_name, nib.load(in_name)))
            dict_metrics_out_name[in_name] = out_name
            measures_output_filename.append(out_name)

    assert_outputs_exist(parser, args, measures_output_filename)
    if not measures_to_compute:
        parser.error('No connectivity measures were selected, nothing '
                     'to compute.')

    logging.info('The following measures will be computed and save: {}'.format(
        measures_output_filename))

    if args.include_dps:
        if not os.path.isdir(args.include_dps):
            os.makedirs(args.include_dps)
        logging.info('data_per_streamline weighting is activated.')

    img_labels = nib.load(args.in_labels)
    data_labels = get_data_as_label(img_labels)
    if not args.force_labels_list:
        labels_list = np.unique(data_labels)[1:].tolist()
    else:
        labels_list = np.loadtxt(
            args.force_labels_list, dtype=np.int16).tolist()

    comb_list = list(itertools.combinations(labels_list, r=2))
    if not args.no_self_connection:
        comb_list.extend(zip(labels_list, labels_list))

    nbr_cpu = validate_nbr_processes(parser, args, args.nbr_processes)
    measures_dict_list = []
    if nbr_cpu == 1:
        for comb in comb_list:
            measures_dict_list.append(_processing_wrapper([args.in_hdf5,
                                                           img_labels, comb,
                                                           measures_to_compute,
                                                           args.similarity,
                                                           args.density_weighting,
                                                           args.include_dps]))
    else:
        pool = multiprocessing.Pool(nbr_cpu)
        measures_dict_list = pool.map(_processing_wrapper,
                                      zip(itertools.repeat(args.in_hdf5),
                                          itertools.repeat(img_labels),
                                          comb_list,
                                          itertools.repeat(
                                              measures_to_compute),
                                          itertools.repeat(args.similarity),
                                          itertools.repeat(
                                          args.density_weighting),
                                          itertools.repeat(args.include_dps)))
        pool.close()
        pool.join()

    # Removing None entries (combinaisons that do not exist)
    # Fusing the multiprocessing output into a single dictionary
    measures_dict_list = [it for it in measures_dict_list if it is not None]
    measures_dict = measures_dict_list[0]
    for dix in measures_dict_list[1:]:
        measures_dict.update(dix)

    if args.no_self_connection:
        total_elem = len(labels_list)**2 - len(labels_list)
        results_elem = len(measures_dict.keys())*2 - len(labels_list)
    else:
        total_elem = len(labels_list)**2
        results_elem = len(measures_dict.keys())*2

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
        elif measure in dict_maps_out_name:
            matrix_basename = dict_maps_out_name[measure]
        else:
            matrix_basename = measure

        np.save(matrix_basename, matrix[:, :, i])


if __name__ == "__main__":
    main()
