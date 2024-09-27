# -*- coding: utf-8 -*-
import copy
import logging
import os
import sys
import threading

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

d = threading.local()


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
    (hdf5_filename, labels_img, comb,
     compute_volume, compute_streamline_count, compute_length,
     similarity_directory, metrics_data, metrics_names, lesion_data,
     include_dps, weighted, min_lesion_vol) = args

    print("Multiprocessing, ID {}: computing info for bundle {}."
          .format(d.id, comb))
    return compute_connectivity_matrices_from_hdf5(
        hdf5_filename, labels_img, comb[0], comb[1],
        compute_volume, compute_streamline_count, compute_length,
        similarity_directory, metrics_data, metrics_names, lesion_data,
        include_dps, weighted, min_lesion_vol)


def compute_connectivity_matrices_from_hdf5(
        hdf5_filename, labels_img, in_label, out_label,
        compute_volume=True, compute_streamline_count=True,
        compute_length=True, similarity_directory=None, metrics_data=None,
        metrics_names=None, lesion_data=None, include_dps=False,
        weighted=False, min_lesion_vol=0):
    """
    Parameters
    ----------
    hdf5_filename: str
        Name of the hdf5 file containing the precomputed connections (bundles)
    labels_img: np.ndarray
        Data as labels
    in_label: str
        Name of one extremity of the bundle to analyse.
    out_label: str
        Name of the other extremity. Current node is {in_label}_{out_label}.
    compute_volume: bool
        If true, return 'volume' in the returned dictionary with the volume of
        the bundle.
    compute_streamline_count: bool
        If true, return 'streamline_count' in the returned dictionary, with
        the number of streamlines in the bundle.
    compute_length: bool
        If true, return 'length' in the returned dictionary, with the mean
        length of streamlines in the bundle.
    similarity_directory: str
        If not None, ??
    metrics: Tuple[list[np.ndarray], list[str]]
        List of 3D data with metrics to use, with the list of associated metric
        names. If set, the returned dictionary will contain an entry for each
        name, with the mean value of each metric.
    lesion_data: Tuple[list, np.ndarray]
        The (lesion_labels, lesion_data) for lesion load analysis. If set, the
        returned dictionary will contain the three entries 'lesion_volume':
        the total lesion volume, 'lesion_streamline_count': the number of
        streamlines passing through lesions, 'lesion_count': the number of
        lesions.
    include_dps: bool
        If true, return an entry for each dps with the mean dps value.
    weighted: bool
        If true, weight the results with the density map.
    min_lesion_vol: float
        Minimum lesion volume for a lesion to be considered.

    Returns
    -------
    final_dict: {(in_label, out_label): (measures_dict, dps_keys)}
        A dictionary with the node as key and as value:
        measures_dict: The dictionary of returned values.
        dps_keys: The list of keys included from dps.
        If the connection is not found, None is returned instead.
    """
    if len(metrics_data) > 0:
        assert len(metrics_data) == len(metrics_names)

    affine, dimensions, voxel_sizes, _ = get_reference_info(labels_img)

    # Getting the bundle from the hdf5
    with h5py.File(hdf5_filename, 'r') as hdf5_file:
        key = '{}_{}'.format(in_label, out_label)
        if key not in hdf5_file:
            logging.debug("Connection {} not found in the hdf5".format(key))
            return None
        streamlines = reconstruct_streamlines_from_hdf5(hdf5_file[key])
        if len(streamlines) == 0:
            logging.debug("Connection {} contained no streamline".format(key))
            return None

    # If density is not required, do not compute it
    # Only required for volume, similarity and any metrics
    if (compute_volume or similarity_directory is not None or
            len(metrics) > 0):
        density = compute_tract_counts_map(streamlines, dimensions)

    measures_to_return = {}

    if compute_length:
        # scil_tractogram_segment_connections_from_labels.py requires
        # isotropic voxels
        mean_length = np.average(length(list(streamlines))) * voxel_sizes[0]
        measures_to_return['length'] = mean_length

    if compute_volume:
        measures_to_return['volume'] = np.count_nonzero(density) * \
                                       np.prod(voxel_sizes)

    if compute_streamline_count:
        measures_to_return['streamline_count'] = len(streamlines)

    if similarity_directory is not None:
        density_sim = load_node_nifti(similarity_directory,
                                      in_label, out_label, labels_img)
        if density_sim is None:
            ba_vox = 0
        else:
            ba_vox = compute_bundle_adjacency_voxel(density, density_sim)

        measures_to_return['similarity'] = ba_vox

    for metric_data, metric_name in zip(metrics_data, metrics_names):
        if weighted:
            avg_value = np.average(metric_data, weights=density)
        else:
            avg_value = np.average(metric_data[density > 0])
        measures_to_return[metric_name] = avg_value

    if lesion_data is not None:
        lesion_labels, lesion_img = lesion_data
        voxel_sizes = lesion_img.header.get_zooms()[0:3]
        lesion_img.set_filename('tmp.nii.gz')
        lesion_atlas = get_data_as_labels(lesion_img)
        tmp_dict = compute_lesion_stats(
            density.astype(bool), lesion_atlas,
            voxel_sizes=voxel_sizes, single_label=True,
            min_lesion_vol=min_lesion_vol,
            precomputed_lesion_labels=lesion_labels)

        tmp_ind = _streamlines_in_mask(list(streamlines),
                                       lesion_atlas.astype(np.uint8),
                                       np.eye(3), [0, 0, 0])
        streamlines_count = len(
            np.where(tmp_ind == [0, 1][True])[0].tolist())

        if tmp_dict:
            measures_to_return['lesion_vol'] = tmp_dict['lesion_total_volume']
            measures_to_return['lesion_count'] = tmp_dict['lesion_count']
            measures_to_return['lesion_streamline_count'] = streamlines_count
        else:
            measures_to_return['lesion_vol'] = 0
            measures_to_return['lesion_count'] = 0
            measures_to_return['lesion_streamline_count'] = 0

    dps_keys = []
    if include_dps:
        for dps_key in hdf5_file[key].keys():
            if dps_key not in ['data', 'offsets', 'lengths']:
                if 'commit' in dps_key:
                    dps_values = np.sum(hdf5_file[key][dps_key])
                else:
                    dps_values = np.average(hdf5_file[key][dps_key])
                measures_to_return[dps_key] = dps_values
                dps_keys.append(dps_key)

    return {(in_label, out_label): measures_to_return}, dps_keys
