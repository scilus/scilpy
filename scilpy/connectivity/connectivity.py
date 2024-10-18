# -*- coding: utf-8 -*-
import copy
import logging
import os
import sys
import threading

from dipy.io.stateful_tractogram import StatefulTractogram
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
    metrics_data: list[np.ndarray]
        List of 3D data with metrics to use, with the list of associated metric
        names. If set, the returned dictionary will contain an entry for each
        name, with the mean value of each metric.
    metrics_names: list[str]
        The metrics names.
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
    final_dict: Tuple[dict, list[str]] or None
        dict: {(in_label, out_label): measures_dict}
            A dictionary with the node as key and as the dictionary as
            described above.
        dps_keys: The list of keys included from dps.
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
            len(metrics_data) > 0):
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


def find_streamlines_with_connectivity(
        streamlines, start_labels, end_labels, label1, label2=None):
    """
    Returns streamlines corresponding to a (label1, label2) or (label2, label1)
    connection.

    Parameters
    ----------
    streamlines: list of np arrays or list of tensors.
        Streamlines, in vox space, corner origin.
    start_labels: list[int]
        The starting bloc for each streamline.
    end_labels: list[int]
        The ending bloc for each streamline.
    label1: int
        The bloc of interest, either as starting or finishing point.
    label2: int, optional
        The bloc of interest, either as starting or finishing point.
        If label2 is None, then all connections (label1, Y) and (X, label1)
        are found.
    """
    start_labels = np.asarray(start_labels)
    end_labels = np.asarray(end_labels)

    if label2 is None:
        labels2 = np.unique(np.concatenate((start_labels[:], end_labels[:])))
    else:
        labels2 = [label2]

    found = np.zeros(len(streamlines))
    for label2 in labels2:
        str_ind1 = np.logical_and(start_labels == label1,
                                  end_labels == label2)
        str_ind2 = np.logical_and(start_labels == label2,
                                  end_labels == label1)
        str_ind = np.logical_or(str_ind1, str_ind2)
        found = np.logical_or(found, str_ind)

    return [s for i, s in enumerate(streamlines) if found[i]]


def compute_triu_connectivity_from_labels(tractogram, data_labels,
                                          hide_background=None):
    """
    Compute a connectivity matrix.

    Parameters
    ----------
    tractogram: StatefulTractogram, or list[np.ndarray]
        Streamlines. A StatefulTractogram input is recommanded.
        When using directly with a list of streamlines, streamlinee must be in
        vox space, corner origin.
    data_labels: np.ndarray
        The loaded nifti image.
    hide_background: Optional[int]
        If not None, streamlines ending in a voxel with given label are
        ignored (i.e. matrix is set to 0 for that label). Suggestion: 0.

    Returns
    -------
    matrix: np.ndarray
        With use_scilpy: shape (nb_labels + 1, nb_labels + 1)
        Else, shape (nb_labels, nb_labels)
    labels: List
        The list of labels. With use_scilpy, last label is: "Not Found".
    start_labels: List
        For each streamline, the label at starting point.
    end_labels: List
        For each streamline, the label at ending point.
    """
    if isinstance(tractogram, StatefulTractogram):
        #  Vox space, corner origin
        # = we can get the nearest neighbor easily.
        # Coord 0 = voxel 0. Coord 0.9 = voxel 0. Coord 1 = voxel 1.
        tractogram.to_vox()
        tractogram.to_corner()
        streamlines = tractogram.streamlines
    else:
        streamlines = tractogram

    real_labels = list(np.sort(np.unique(data_labels)))
    nb_labels = len(real_labels)
    logging.debug("Computing connectivity matrix for {} labels."
                  .format(nb_labels))

    matrix = np.zeros((nb_labels, nb_labels), dtype=int)
    start_labels = []
    end_labels = []

    for s in streamlines:
        start = real_labels.index(
            data_labels[tuple(np.floor(s[0, :]).astype(int))])
        end = real_labels.index(
            data_labels[tuple(np.floor(s[-1, :]).astype(int))])

        start_labels.append(start)
        end_labels.append(end)

        matrix[start, end] += 1
        if start != end:
            matrix[end, start] += 1

    matrix = np.triu(matrix)
    assert matrix.sum() == len(streamlines)

    if hide_background is not None:
        idx = real_labels.index(hide_background)
        nb_hidden = np.sum(matrix[idx, :]) + np.sum(matrix[:, idx]) - \
            matrix[idx, idx]
        if nb_hidden > 0:
            logging.warning("CAREFUL! {} streamlines had one or both "
                            "endpoints in a non-labelled area "
                            "(background = label {}; line/column {})"
                            .format(nb_hidden, hide_background, idx))
            matrix[idx, :] = 0
            matrix[:, idx] = 0
        else:
            logging.info("No streamlines with endpoints in the background :)")
        real_labels[idx] = ("Hidden background ({})".format(hide_background))

    return matrix, real_labels, start_labels, end_labels
