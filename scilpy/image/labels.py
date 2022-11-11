# -*- coding: utf-8 -*-
import logging
import os

import numpy as np


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


def combine_labels(data_list, indices_per_input_volume, out_labels_choice,
                   background_id=0, merge_groups=False):
    """
    - Output options: Only one of out_labels, unique or group_in_m may be
      chosen.
    - Merge_groups can only be used in group_in_m.

    Parameters
    ----------
    data_list: list
        List of np.ndarray. Data as labels.
    indices_per_input_volume: list[np.ndarray]
        List of np.ndarray containing the indices to use in each input volume.
    out_labels_choice: tuple(str, any)
        Tuple of a string expressing the choice of output option and the
        associated necessary value. Choices are:
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
