# -*- coding: utf-8 -*-
import logging
import os
import threading

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.utils import is_header_compatible, get_reference_info
from dipy.tracking.streamlinespeed import length
from dipy.tracking.vox2track import _streamlines_in_mask
import h5py
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from scilpy.image.labels import get_data_as_labels
from scilpy.io.hdf5 import reconstruct_streamlines_from_hdf5
from scilpy.tractanalysis.reproducibility_measures import \
    compute_bundle_adjacency_voxel
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractograms.streamline_operations import \
    resample_streamlines_num_points
from scilpy.utils.metrics_tools import compute_lesion_stats


d = threading.local()


def compute_triu_connectivity_from_labels(tractogram, data_labels,
                                          keep_background=False,
                                          hide_labels=None):
    """
    Compute a connectivity matrix.

    Parameters
    ----------
    tractogram: StatefulTractogram, or list[np.ndarray]
        Streamlines. A StatefulTractogram input is recommanded.
        When using directly with a list of streamlines, streamlines must be in
        vox space, center origin.
    data_labels: np.ndarray
        The loaded nifti image.
    keep_background: Bool
        By default, the background (label 0) is not included in the matrix.
        If True, label 0 is kept.
    hide_labels: Optional[List[int]]
        If not None, streamlines ending in a voxel with a given label are
        ignored (i.e. matrix is set to 0 for that label).

    Returns
    -------
    matrix: np.ndarray
        With use_scilpy: shape (nb_labels + 1, nb_labels + 1)
        Else, shape (nb_labels, nb_labels)
    ordered_labels: List
        The list of labels. Name of each row / column.
    start_labels: List
        For each streamline, the label at starting point.
    end_labels: List
        For each streamline, the label at ending point.
    """
    if isinstance(tractogram, StatefulTractogram):
        # vox space, center origin: compatible with map_coordinates
        sfs_2_pts = resample_streamlines_num_points(tractogram, 2)
        sfs_2_pts.to_vox()
        sfs_2_pts.to_center()
        streamlines = sfs_2_pts.streamlines

    else:
        streamlines = tractogram

    ordered_labels = list(np.sort(np.unique(data_labels)))
    assert ordered_labels[0] >= 0, "Only accepting positive labels."
    nb_labels = len(ordered_labels)
    logging.debug("Computing connectivity matrix for {} labels."
                  .format(nb_labels))

    matrix = np.zeros((nb_labels, nb_labels), dtype=int)

    labels = map_coordinates(data_labels, streamlines._data.T, order=0)
    start_labels = labels[0::2]
    end_labels = labels[1::2]

    # sort each pair of labels for start to be smaller than end
    start_labels, end_labels = zip(*[sorted(pair) for pair in
                                     zip(start_labels, end_labels)])

    np.add.at(matrix, (start_labels, end_labels), 1)
    assert matrix.sum() == len(streamlines)

    # Rejecting background
    if not keep_background and ordered_labels[0] == 0:
        logging.debug("Rejecting background.")
        ordered_labels = ordered_labels[1:]
        matrix = matrix[1:, 1:]

    # Hiding labels
    if hide_labels is not None:
        for label in hide_labels:
            if label not in ordered_labels:
                logging.warning("Cannot hide label {} because it was not in "
                                "the data.".format(label))
                continue
            idx = ordered_labels.index(label)
            nb_hidden = np.sum(matrix[idx, :]) + np.sum(matrix[:, idx]) - \
                matrix[idx, idx]
            if nb_hidden > 0:
                logging.warning("{} streamlines had one or both endpoints "
                                "in hidden label {} (line/column {})"
                                .format(nb_hidden, label, idx))
                matrix[idx, :] = 0
                matrix[:, idx] = 0
            else:
                logging.info("No streamlines with endpoints in hidden label "
                             "{} (line/column {}) :)".format(label, idx))
            ordered_labels[idx] = ("Hidden label ({})".format(label))

    return matrix, ordered_labels, start_labels, end_labels


def load_node_nifti(directory, in_label, out_label, ref_img):
    in_filename = os.path.join(directory,
                               '{}_{}.nii.gz'.format(in_label, out_label))

    if os.path.isfile(in_filename):
        if not is_header_compatible(in_filename, ref_img):
            raise IOError('{} do not have a compatible header'.format(
                in_filename))
        return nib.load(in_filename).get_fdata(dtype=np.float32)

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
        If true, return 'volume_mm3' in the returned dictionary with the volume
        of the bundle.
    compute_streamline_count: bool
        If true, return 'streamline_count' in the returned dictionary, with
        the number of streamlines in the bundle.
    compute_length: bool
        If true, return 'length_mm' in the returned dictionary, with the mean
        length of streamlines in the bundle.
    similarity_directory: str
        If not None, should be a directory containing nifti files that
        represent density maps for each connection, using the
        <in_label>_<out_label>.nii.gz conventions.
        Typically computed from a template (must be in the same space).
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

    measures_to_return = {}

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
        logging.debug("Found {} streamlines for connection {}"
                      .format(len(streamlines), key))

        # Getting dps info from the hdf5
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

    # If density is not required, do not compute it
    # Only required for volume, similarity and any metrics
    if (compute_volume or similarity_directory is not None or
            len(metrics_data) > 0):
        density = compute_tract_counts_map(streamlines, dimensions)

    if compute_length:
        # scil_tractogram_segment_connections_from_labels.py requires
        # isotropic voxels
        mean_length = np.average(length(list(streamlines))) * voxel_sizes[0]
        measures_to_return['length_mm'] = mean_length

    if compute_volume:
        measures_to_return['volume_mm3'] = np.count_nonzero(density) * \
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

    return {(in_label, out_label): measures_to_return}, dps_keys
