# -*- coding: utf-8 -*-
from importlib.resources import files
import json
import logging
import os
import tqdm

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from scilpy.tractanalysis.reproducibility_measures import compute_bundle_adjacency_voxel

SCILPY_LUT_DIR = files("scilpy").joinpath("data/LUT/")


def load_wmparc_labels():
    """
    Load labels dictionary of different parcellations from the Desikan-Killiany
    atlas.
    """
    labels_path = SCILPY_LUT_DIR.joinpath('dk_aggregate_structures.json')
    with open(labels_path) as labels_file:
        labels = json.load(labels_file)
    return labels


def get_data_as_labels(in_img):
    """
    Get data as label (force type np.uint16), check data type before casting.

    Parameters
    ----------
    in_img: nibabel.nifti1.Nifti1Image
        Image.

    Return
    ------
    data: numpy.ndarray
        Data (dtype: np.uint16).
    """
    curr_type = in_img.get_data_dtype()

    if np.issubdtype(curr_type, np.signedinteger) or \
            np.issubdtype(curr_type, np.unsignedinteger):
        return np.asanyarray(in_img.dataobj).astype(np.uint16)
    else:
        basename = os.path.basename(in_img.get_filename())
        raise IOError('The image {} cannot be loaded as label because '
                      'its format {} is not compatible with a label '
                      'image'.format(basename, curr_type))


def get_binary_mask_from_labels(atlas, label_list):
    """
    Get a binary mask from labels.

    Parameters
    ----------
    atlas: numpy.ndarray
        The image will all labels as values (ex, result from
        get_data_as_labels).
    label_list: list[int]
        The labels to get.
    """
    mask = np.zeros(atlas.shape, dtype=np.uint16)
    for label in label_list:
        is_label = atlas == label
        mask[is_label] = 1

    return mask


def get_labels_from_mask(mask_data, labels=None, background_label=0,
                         min_voxel_count=0):
    """
    Get labels from a binary mask which contains multiple blobs. Each blob
    will be assigned a label, by default starting from 1. Background will
    be assigned the background_label value.

    Parameters
    ----------
    mask_data: np.ndarray
        The mask data.
    labels: list, optional
        Labels to assign to each blobs in the mask. Excludes the background
        label.
    background_label: int
        Label for the background.
    min_voxel_count: int, optional
        Minimum number of voxels for a blob to be considered. Blobs with fewer
        voxels will be ignored.

    Returns
    -------
    label_map: np.ndarray
        The labels.
    """
    # Get the number of structures and assign labels to each blob
    label_map, nb_structures = ndi.label(mask_data)
    if min_voxel_count:
        new_count = 0
        for label in range(1, nb_structures + 1):
            if np.count_nonzero(label_map == label) < min_voxel_count:
                label_map[label_map == label] = 0
            else:
                new_count += 1
                label_map[label_map == label] = new_count
        logging.debug(
            f"Ignored blob {nb_structures-new_count} with fewer "
            "than {min_voxel_count} voxels")
        nb_structures = new_count

    # Assign labels to each blob if provided
    if labels:
        # Only keep the first nb_structures labels if the number of labels
        # provided is greater than the number of blobs in the mask.
        if len(labels) > nb_structures:
            logging.warning("Number of labels ({}) does not match the number "
                            "of blobs in the mask ({}). Only the first {} "
                            "labels will be used.".format(
                                len(labels), nb_structures, nb_structures))
        # Cannot assign fewer labels than the number of blobs in the mask.
        elif len(labels) < nb_structures:
            raise ValueError("Number of labels ({}) is less than the number of"
                             " blobs in the mask ({}).".format(
                                 len(labels), nb_structures))

        # Copy the label map to avoid scenarios where the label list contains
        # labels that are already present in the label map
        custom_label_map = label_map.copy()
        # Assign labels to each blob
        for idx, label in enumerate(labels[:nb_structures]):
            custom_label_map[label_map == idx + 1] = label
        label_map = custom_label_map

    logging.info('Assigned labels {} to the mask.'.format(
        np.unique(label_map[label_map != background_label])))

    if background_label != 0 and background_label in label_map:
        logging.warning("Background label {} corresponds to a label "
                        "already in the map. This will cause issues.".format(
                            background_label))

    # Assign background label
    if background_label:
        label_map[label_map == 0] = background_label

    return label_map


def split_labels(labels_volume, label_indices):
    """
    For each label in list, return a separate volume containing only that
    label.

    Parameters
    ----------
    labels_volume: np.ndarray
        A 3D volume.
    label_indices: list or np.array
        The list of labels to extract.

    Returns
    -------
    split_data: list
        One 3D volume per label.
    """
    split_data = []
    for label in label_indices:
        label_occurrences = np.where(labels_volume == int(label))
        if len(label_occurrences) != 0:
            split_label = np.zeros(labels_volume.shape, dtype=np.uint16)
            split_label[label_occurrences] = label
            split_data.append(split_label)
        else:
            logging.info("Label {} not present in the image.".format(label))
            split_data.append(None)
    return split_data


def remove_labels(labels_volume, label_indices, background_id=0):
    """
    Remove given labels from the volume.

    Parameters
    ----------
    labels_volume: np.ndarray
        The volume (as labels).
    label_indices: list
        List of labels indices to remove.
    background_id: int
        Value used for removed labels
    """
    for index in np.unique(label_indices):
        mask = labels_volume == index
        labels_volume[mask] = background_id
        if np.count_nonzero(mask) == 0:
            logging.warning("Label {} was not in the volume".format(index))
    return labels_volume


def combine_labels(data_list, indices_per_input_volume, out_labels_choice,
                   background_id=0, merge_groups=False):
    """

    Parameters
    ----------
    data_list: list
        List of np.ndarray. Data as labels.
    indices_per_input_volume: list[np.ndarray]
        List of np.ndarray containing the indices to use in each input volume.
    out_labels_choice: tuple(str, any)
        Tuple of a string expressing the choice of output option and the
        associated necessary value.
        Choices are:

        ('all_labels'): Keeps values from the input volumes, or with
            merge_groups, used the volumes ordering.

        ('out_label_ids', list): Out labels will be renamed as given from
            the list.

        ('unique'): Out labels will be renamed to range from 1 to
            total_nb_labels (+ the background).

        ('group_in_m'): Add (x * 10 000) to each volume labels, where x is the
            input volume order number.
    background_id: int
        Background id, excluded from output. The value is also used as output
        background value. Default : 0.
    merge_groups: bool
        If true, indices from indices_per_input_volume will be merged for each
        volume, as a single label. Can only be used with 'all_labels' or
        'out_label_ids'. Default: False.
    """
    assert out_labels_choice[0] in ['all_labels', 'out_labels_ids',
                                    'unique', 'group_in_m']
    if merge_groups and out_labels_choice[0] in ['unique', 'group_in_m']:
        raise ValueError("Merge groups cannot be used together with "
                         "'unique' in 'group_in_m options.")

    nb_volumes = len(data_list)

    filtered_ids_per_vol = []
    total_nb_input_ids = 0
    # Remove background labels
    for id_list in indices_per_input_volume:
        id_list = np.asarray(id_list)
        new_ids = id_list[~np.in1d(id_list, background_id)]
        filtered_ids_per_vol.append(new_ids)
        total_nb_input_ids += len(new_ids)

    # Prepare output ids.
    if out_labels_choice[0] == 'out_labels_ids':
        out_labels = out_labels_choice[1]
        if merge_groups:
            assert len(out_labels) == nb_volumes, \
                "Expecting {} output labels.".format(nb_volumes)
        else:
            assert len(out_labels) == total_nb_input_ids
    elif out_labels_choice[0] == 'unique':
        stack = np.hstack(filtered_ids_per_vol)
        ids = np.arange(len(stack) + 1)
        out_labels = np.setdiff1d(ids, background_id)[:len(stack)]
    elif out_labels_choice[0] == 'group_in_m':
        m_list = []
        for i in range(len(filtered_ids_per_vol)):
            prefix = i * 10000
            m_list.append(prefix + np.asarray(filtered_ids_per_vol[i]))
        out_labels = np.hstack(m_list)
    else:  # all_labels
        if merge_groups:
            out_labels = np.arange(nb_volumes) + 1
        else:
            out_labels = np.hstack(filtered_ids_per_vol)

    if len(np.unique(out_labels)) != len(out_labels):
        logging.warning("The same output label number will be used for "
                        "multiple inputs!")

    # Create the resulting volume
    current_id = 0
    resulting_labels = (np.ones_like(data_list[0], dtype=np.uint16)
                        * background_id)
    for i in range(nb_volumes):
        # Loop on ids for this volume.
        for this_id in filtered_ids_per_vol[i]:
            mask = data_list[i] == this_id
            if np.count_nonzero(mask) == 0:
                logging.warning(
                    "Label {} was not in the volume".format(this_id))

            if merge_groups:
                resulting_labels[mask] = out_labels[i]
            else:
                resulting_labels[mask] = out_labels[current_id]
                current_id += 1

    return resulting_labels


def dilate_labels(data, vox_size, distance, nbr_processes,
                  labels_to_dilate=None, labels_not_to_dilate=None,
                  labels_to_fill=None, mask=None):
    """
    Parameters
    ----------
    data: np.ndarray
        The data (as labels) to dilate.
    vox_size: np.ndarray(1, 3)
        The voxel size.
    distance: float
        Maximal distance to dilate (in mm).
    nbr_processes: int
        Number of processes.
    labels_to_dilate: list, optional
        Label list to dilate. By default it dilates all labels not in
        labels_to_fill nor in labels_not_to_dilate.
    labels_not_to_dilate: list, optional
        Label list not to dilate.
    labels_to_fill: list, optional
        Background id / labels to be filled. The first one is given as output
        background value. Default: [0]
    mask: np.ndarray, optional
        Only dilate values inside the mask.
    """
    if labels_to_fill is None:
        labels_to_fill = [0]

    img_shape = data.shape

    # Check if in both: label_to_fill & not_to_fill
    fill_and_not = np.intersect1d(labels_not_to_dilate, labels_to_fill)
    if len(fill_and_not) > 0:
        logging.error("Error, both in not_to_dilate and to_fill: {}".format(
            np.asarray(labels_not_to_dilate)[fill_and_not]))

    # Create background mask
    is_background_mask = np.zeros(img_shape, dtype=bool)
    for i in labels_to_fill:
        is_background_mask = np.logical_or(is_background_mask, data == i)

    # Create not_to_dilate mask (initialized to background)
    not_to_dilate = np.copy(is_background_mask)
    for i in labels_not_to_dilate:
        not_to_dilate = np.logical_or(not_to_dilate, data == i)

    # Add mask
    if mask is not None:
        to_dilate_mask = np.logical_and(is_background_mask, mask)
    else:
        to_dilate_mask = is_background_mask

    # Create label mask
    is_label_mask = ~not_to_dilate

    if labels_to_dilate is not None:
        # Check if in both: to_dilate & not_to_dilate
        dil_and_not = np.in1d(labels_to_dilate, labels_not_to_dilate)
        if np.any(dil_and_not):
            logging.error("Error, both in dilate and Not to dilate: {}".format(
                np.asarray(labels_to_dilate)[dil_and_not]))

        # Check if in both: to_dilate & to_fill
        dil_and_fill = np.in1d(labels_to_dilate, labels_to_fill)
        if np.any(dil_and_fill):
            logging.error("Error, both in dilate and to fill: {}".format(
                np.asarray(labels_to_dilate)[dil_and_fill]))

        # Create new label to dilate list
        new_label_mask = np.zeros_like(data, dtype=bool)
        for i in labels_to_dilate:
            new_label_mask = np.logical_or(new_label_mask, data == i)

        # Combine both new_label_mask and not_to_dilate
        is_label_mask = np.logical_and(new_label_mask, ~not_to_dilate)

    # Get the list of indices
    background_pos = np.argwhere(to_dilate_mask) * vox_size
    label_pos = np.argwhere(is_label_mask) * vox_size
    ckd_tree = cKDTree(label_pos)

    # Compute the nearest labels for each voxel of the background
    dist, indices = ckd_tree.query(
        background_pos, k=1, distance_upper_bound=distance,
        workers=nbr_processes)

    # Associate indices to the nearest label (in distance)
    valid_nearest = np.squeeze(np.isfinite(dist))
    id_background = np.flatnonzero(to_dilate_mask)[valid_nearest]
    id_label = np.flatnonzero(is_label_mask)[indices[valid_nearest]]

    # Change values of those background
    data = data.flatten()
    data[id_background.T] = data[id_label.T]
    data = data.reshape(img_shape)

    return data


def get_stats_in_label(map_data, label_data, label_lut):
    """
    Get statistics about a map for each label in an atlas.

    Parameters
    ----------
    map_data: np.ndarray
        The map from which to get statistics.
    label_data: np.ndarray
        The loaded atlas.
    label_lut: dict
        The loaded label LUT (look-up table).

    Returns
    -------
    out_dict: dict
        A dict with one key per label name, and its values are the computed
        statistics.
    """
    (label_indices, label_names) = zip(*label_lut.items())

    out_dict = {}
    for label, name in zip(label_indices, label_names):
        label = int(label)
        if label != 0:
            curr_data = (map_data[np.where(label_data == label)])
            nb_vx_roi = np.count_nonzero(label_data == label)
            nb_seed_vx = np.count_nonzero(curr_data)

            if nb_seed_vx != 0:
                mean_seed = np.sum(curr_data) / nb_seed_vx
                max_seed = np.max(curr_data)
                std_seed = np.sqrt(np.mean(abs(curr_data[curr_data != 0] -
                                               mean_seed) ** 2))

                out_dict[name] = {'ROI-idx': label,
                                  'ROI-name': str(name),
                                  'nb-vx-roi': int(nb_vx_roi),
                                  'nb-vx-seed': int(nb_seed_vx),
                                  'max': int(max_seed),
                                  'mean': float(mean_seed),
                                  'std': float(std_seed)}
    return out_dict


def merge_labels_into_mask(atlas, filtering_args):
    """
    Merge labels into a mask.

    Parameters
    ----------
    atlas: np.ndarray
        Atlas with labels as a numpy array (uint16) to merge.

    filtering_args: str
        Filtering arguments from the command line.

    Return
    ------
    mask: nibabel.nifti1.Nifti1Image
        Mask obtained from the combination of multiple labels.
    """
    mask = np.zeros(atlas.shape, dtype=np.uint16)

    if ' ' in filtering_args:
        values = filtering_args.split(' ')
        for filter_opt in values:
            if ':' in filter_opt:
                vals = [int(x) for x in filter_opt.split(':')]
                mask[(atlas >= int(min(vals))) & (atlas <= int(max(vals)))] = 1
            else:
                mask[atlas == int(filter_opt)] = 1
    elif ':' in filtering_args:
        values = [int(x) for x in filtering_args.split(':')]
        mask[(atlas >= int(min(values))) & (atlas <= int(max(values)))] = 1
    else:
        mask[atlas == int(filtering_args)] = 1

    return mask


def harmonize_labels(original_data, min_voxel_overlap=1, max_adjacency=1e2):
    """
    Harmonize lesion labels across multiple 3D volumes by ensuring consistent
    labeling.

    This function takes multiple 3D NIfTI volumes with labeled regions
    (e.g., lesions) and harmonizes the labels so that regions that are the same
    across different volumes are assigned a consistent label. It operates by
    iteratively comparing labels in each volume to those in previous volumes
    and matching them based on spatial proximity and overlap.

    Parameters
    ----------
    original_data : list of numpy.ndarray
        A list of 3D numpy arrays where each array contains labeled regions.
        Labels should be non-zero integers, where each unique integer represents
        a different region or lesion.
    min_voxel_overlap : int, optional
        Minimum number of overlapping voxels required for two regions (lesions)
        from different volumes to be considered as potentially the same lesion.
        Default is 1.
    max_adjacency : float, optional
        Maximum distance allowed between the centroids of two regions for them
        to be considered as the same lesion. Default is 1e2 (infinite).

    Returns
    -------
    list of numpy.ndarray
        A list of 3D numpy arrays with the same shape as `original_data`, where
        labels have been harmonized across all volumes. Each region across
        volumes that is identified as the same will have the same label.
    """

    relabeled_data = [np.zeros_like(data) for data in original_data]
    relabeled_data[0] = original_data[0]
    labels = np.unique(original_data)[1:]

    # We will iterate over all possible combinations of labels
    N = len(original_data)
    total_iteration = ((N * (N - 1)) // 2)
    tqdm_bar = tqdm.tqdm(total=total_iteration, desc="Harmonizing labels")

    # We want to label images in order
    for first_pass in range(len(original_data)):
        unmatched_labels = np.unique(original_data[first_pass])[1:].tolist()
        best_match_score = {label: 999999 for label in labels}
        best_match_pos = {label: None for label in labels}

        # We iterate over all previous images to find the best match
        for second_pass in range(0, first_pass):
            tqdm_bar.update(1)

            # We check all existing labels in relabeled data
            for label_ind_1 in range(len(labels)):
                label_1 = labels[label_ind_1]

                if label_1 not in original_data[first_pass]:
                    continue

                # This check requires to at least overlap by N voxel
                coord_1 = np.where(original_data[first_pass] == label_1)
                overlap_labels_count = np.unique(relabeled_data[second_pass][coord_1],
                                                 return_counts=True)

                potential_labels_val = overlap_labels_count[0].tolist()
                potential_labels_count = overlap_labels_count[1].tolist()
                potential_labels = []
                for val, count in zip(potential_labels_val,
                                      potential_labels_count):
                    if val != 0 and count > min_voxel_overlap:
                        potential_labels.append(val)

                # We check all labels touching the previous label
                for label_2 in potential_labels:
                    tmp_data_1 = np.zeros_like(original_data[0])
                    tmp_data_2 = np.zeros_like(original_data[0])

                    # We always compare the previous relabeled data with the next
                    # original data
                    tmp_data_1[original_data[first_pass] == label_1] = 1
                    tmp_data_2[relabeled_data[second_pass] == label_2] = 1

                    # They should have a similar shape (TODO: parameters)
                    adjacency = compute_bundle_adjacency_voxel(
                        tmp_data_1, tmp_data_2)
                    if adjacency > max_adjacency:
                        continue

                    if adjacency < best_match_score[label_1]:
                        best_match_score[label_1] = adjacency
                        best_match_pos[label_1] = label_2

            # We relabel the data and keep track of the unmatched labels
            for label in labels:
                if best_match_pos[label] is not None:
                    old_label = label
                    new_label = best_match_pos[label]
                    relabeled_data[first_pass][original_data[first_pass]
                                               == old_label] = new_label
                    if old_label in unmatched_labels:
                        unmatched_labels.remove(old_label)

        # Anything that is left should be given a new label
        if first_pass == 0:
            continue
        next_label = np.max(relabeled_data[:first_pass]) + 1
        for label in unmatched_labels:
            relabeled_data[first_pass][original_data[first_pass]
                                       == label] = next_label
            next_label += 1

    return [data.astype(np.uint16) for data in relabeled_data]
