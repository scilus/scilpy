# -*- coding: utf-8 -*-
import inspect
import logging
import os

import numpy as np
from scipy.spatial import cKDTree


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


def get_lut_dir():
    """
    Return LUT directory in scilpy repository

    Returns
    -------
    lut_dir: string
        LUT path
    """
    # Get the valid LUT choices.
    import scilpy  # ToDo. Is this the only way?
    module_path = inspect.getfile(scilpy)

    lut_dir = os.path.join(os.path.dirname(
        os.path.dirname(module_path)) + "/data/LUT/")

    return lut_dir


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
        label_occurences = np.where(labels_volume == int(label))
        if len(label_occurences) != 0:
            split_label = np.zeros(labels_volume.shape, dtype=np.uint16)
            split_label[label_occurences] = label
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
