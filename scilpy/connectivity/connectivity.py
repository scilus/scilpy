# -*- coding: utf-8 -*-
import copy
import logging
import os

from dipy.io.utils import is_header_compatible, get_reference_info
from dipy.tracking.streamlinespeed import length
from dipy.tracking.vox2track import _streamlines_in_mask
import h5py
import nibabel as nib
import numpy as np

from scilpy.image.labels import get_data_as_labels
from scilpy.io.hdf5 import (assert_header_compatible_hdf5,
                            reconstruct_streamlines_from_hdf5)
from scilpy.tractanalysis.reproducibility_measures import \
    compute_bundle_adjacency_voxel
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.utils.metrics_tools import compute_lesion_stats


def load_node_nifti(directory, in_label, out_label, ref_img):
    in_filename = os.path.join(directory,
                               '{}_{}.nii.gz'.format(in_label, out_label))

    if os.path.isfile(in_filename):
        if not is_header_compatible(in_filename, ref_img):
            raise IOError('{} do not have a compatible header'.format(
                in_filename))
        return nib.load(in_filename).get_fdata(dtype=np.float64)

    return None


def multi_proc_compute_connectivity_matrices_from_hdf5(args):
    hdf5_filename = args[0]
    labels_img = args[1]
    comb = args[2]
    measures_to_compute = copy.copy(args[3])
    similarity = args[4]
    weighted = args[5]
    include_dps = args[6]
    min_lesion_vol = args[7]
    return compute_connectivity_matrices_from_hdf5(
        hdf5_filename, labels_img, comb, measures_to_compute,
        similarity, weighted, include_dps, min_lesion_vol)


def compute_connectivity_matrices_from_hdf5(
        hdf5_filename, labels_img, comb, measures_to_compute,
        similarity, weighted, include_dps, min_lesion_vol):
    if similarity is not None:
        similarity_directory = similarity[0]

    in_label, out_label = comb
    hdf5_file = h5py.File(hdf5_filename, 'r')
    key = '{}_{}'.format(in_label, out_label)
    if key not in hdf5_file:
        return
    streamlines = reconstruct_streamlines_from_hdf5(hdf5_file[key])
    if len(streamlines) == 0:
        return

    affine, dimensions, voxel_sizes, _ = get_reference_info(labels_img)
    measures_to_return = {}
    assert_header_compatible_hdf5(hdf5_file, (affine, dimensions))

    # Precompute to save one transformation, insert later
    if 'length' in measures_to_compute:
        streamlines_copy = list(streamlines)
        # scil_tractogram_segment_connections_from_labels.py requires
        # isotropic voxels
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
        # Maps
        if isinstance(measure, str) and os.path.isdir(measure):
            map_dirname = measure
            map_data = load_node_nifti(map_dirname,
                                       in_label, out_label,
                                       labels_img)
            measures_to_return[map_dirname] = np.average(
                map_data[map_data > 0])
        elif isinstance(measure, tuple):
            if not isinstance(measure[0], tuple) \
                    and os.path.isfile(measure[0]):
                metric_filename = measure[0]
                metric_img = measure[1]
                if not is_header_compatible(metric_img, labels_img):
                    logging.error('{} do not have a compatible header'.format(
                        metric_filename))
                    raise IOError

                metric_data = metric_img.get_fdata(dtype=np.float64)
                if weighted:
                    avg_value = np.average(metric_data, weights=density)
                else:
                    avg_value = np.average(metric_data[density > 0])
                measures_to_return[metric_filename] = avg_value
            # lesion
            else:
                lesion_filename = measure[0][0]
                computed_lesion_labels = measure[0][1]
                lesion_img = measure[1]
                if not is_header_compatible(lesion_img, labels_img):
                    logging.error('{} do not have a compatible header'.format(
                        lesion_filename))
                    raise IOError

                voxel_sizes = lesion_img.header.get_zooms()[0:3]
                lesion_img.set_filename('tmp.nii.gz')
                lesion_atlas = get_data_as_labels(lesion_img)
                tmp_dict = compute_lesion_stats(
                    density.astype(bool), lesion_atlas,
                    voxel_sizes=voxel_sizes, single_label=True,
                    min_lesion_vol=min_lesion_vol,
                    precomputed_lesion_labels=computed_lesion_labels)

                tmp_ind = _streamlines_in_mask(list(streamlines),
                                               lesion_atlas.astype(np.uint8),
                                               np.eye(3), [0, 0, 0])
                streamlines_count = len(
                    np.where(tmp_ind == [0, 1][True])[0].tolist())

                if tmp_dict:
                    measures_to_return[lesion_filename+'vol'] = \
                        tmp_dict['lesion_total_volume']
                    measures_to_return[lesion_filename+'count'] = \
                        tmp_dict['lesion_count']
                    measures_to_return[lesion_filename+'sc'] = \
                        streamlines_count
                else:
                    measures_to_return[lesion_filename+'vol'] = 0
                    measures_to_return[lesion_filename+'count'] = 0
                    measures_to_return[lesion_filename+'sc'] = 0

    if include_dps:
        for dps_key in hdf5_file[key].keys():
            if dps_key not in ['data', 'offsets', 'lengths']:
                out_file = os.path.join(include_dps, dps_key)
                if 'commit' in dps_key:
                    measures_to_return[out_file] = np.sum(
                        hdf5_file[key][dps_key])
                else:
                    measures_to_return[out_file] = np.average(
                        hdf5_file[key][dps_key])

    return {(in_label, out_label): measures_to_return}
