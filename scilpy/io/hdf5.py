# -*- coding: utf-8 -*-

import logging

from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
import nibabel as nib
from dipy.io.utils import create_nifti_header
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np

from scilpy.io.streamlines import reconstruct_streamlines


def reconstruct_sft_from_hdf5(hdf5_handle, group_keys, space=Space.VOX,
                              origin=Origin.TRACKVIS, load_dps=False,
                              load_dpp=False, ref_img=None,
                              merge_groups=False, allow_empty=False):
    """
    Reconstructs one or more SFT from the HDF5 data.

    Parameters
    ----------
    hdf5_handle: hdf5.file
        Opened hdf5 file.
    group_keys: str or list or None
        Name of the streamlines group(s) in the HDF5. If None, loads all
        groups. If more than one group, by default, will return one sft per
        group.
    space: Space
        The space in which the HDF5 was saved. Default: Space.Vox
    origin: Origin
        The origin in which the HDF5 was saved. Default: Origin.TRACKVIS
    load_dps: bool
        If true, loads the data_per_streamline if present in the hdf5.
    load_dpp: bool
        If true, loads the data_per_point if present in the hdf5.
    ref_img: nib.Nifti1Image
        If set, will verify that the HDF5's header is compatible with this
        reference.
    merge_groups: bool
        If true, and if groups_keys refer to more than one group, will merge
        all bundles into one SFT. Else, returns one SFT per group.
    allow_empty: bool
        If true, if no streamlines are found, an empty tractogram will be
        returned. If one or more group_keys do not exist, no error will be
        raised.

    Returns
    -------
    sft: StatefulTractogram or list[StatefulTractogram]
        The tractogram(s)
    groups_len: list[int]
        The number of streamlines per loaded group.
    """
    if ref_img is not None:
        assert_header_compatible_hdf5(hdf5_handle, ref_img)

    # Prepare header
    affine = hdf5_handle.attrs['affine']
    dimensions = hdf5_handle.attrs['dimensions']
    voxel_sizes = hdf5_handle.attrs['voxel_sizes']
    header = create_nifti_header(affine, dimensions, voxel_sizes)

    # Find the groups to load
    if isinstance(group_keys, str):
        group_keys = [group_keys]
    elif group_keys is None:
        group_keys = list(hdf5_handle.keys())  # Get all groups

    if not isinstance(group_keys, list):
        raise ValueError("Expecting key to be either a str, a list or None. "
                         "(our fault. Code needs revision).")
    if len(group_keys) == 1:
        merge_groups = True

    # Get streamlines. They have been stored inside a group (or one per
    # bundle). Load all groups and either merge together or return many sft.
    groups_len = []
    streamlines = []
    dps = []
    for i, group_key in enumerate(group_keys):
        # Get streamlines
        if group_key not in hdf5_handle:
            if allow_empty:
                tmp_streamlines = []
            else:
                raise ValueError("Group key {} not found in the hdf5. "
                                 "Possible choices: {}"
                                 .format(group_key, list(hdf5_handle.keys())))
        else:
            # If key exists, tmp_streamlines should not be empty.
            tmp_streamlines = reconstruct_streamlines_from_hdf5(
                hdf5_handle[group_key])

        if merge_groups:
            streamlines.extend(tmp_streamlines)
            groups_len.append(len(tmp_streamlines))
        else:
            streamlines.append(tmp_streamlines)

        # Load dps / dpp
        if i == 0 or not merge_groups:
            dps.append({})
        if len(tmp_streamlines) > 0:
            for sub_key in hdf5_handle[group_key].keys():
                if sub_key not in ['data', 'offsets', 'lengths']:
                    data = hdf5_handle[group_key][sub_key]
                    if data.shape == hdf5_handle[group_key]['offsets']:
                        # Discovered dps
                        if load_dps:
                            if i == 0 or not merge_groups:
                                dps[i][sub_key] = data
                            else:
                                dps[i][sub_key] = np.concatenate(
                                    (dps[i][sub_key], data))
                    else:
                        if load_dpp:
                            raise NotImplementedError(
                                "Don't know how to load dpp yet.")
                        raise NotImplementedError("DPP data? Other?")

    # 3) Format as SFT
    if merge_groups:
        if len(streamlines) == 0 and not allow_empty:
            raise ValueError("Cannot load an empty tractogram from HDF5. Set "
                             "`allow_empty` to True if you want to force it.")
        sft = StatefulTractogram(streamlines, header, space=space,
                                 origin=origin, data_per_streamline=dps[0])
        return sft, groups_len
    else:
        sfts = []
        for (sub_streamlines, sub_dps) in zip(streamlines, dps):
            if len(streamlines) == 0 and not allow_empty:
                raise ValueError("Cannot load an empty tractogram from HDF5. "
                                 "Set `allow_empty` to True if you want to "
                                 "force it.")
            else:
                sfts.append(
                    StatefulTractogram(sub_streamlines, header, space=space,
                                       origin=origin,
                                       data_per_streamline=sub_dps))
        return sfts, groups_len


def assert_header_compatible_hdf5(hdf5_handle, ref):
    """
    Parameters
    ----------
    hdf5_handle: hdf5.file
        Opened hdf5 handle.
    ref: Either a tuple (affine, dimension) or a nib.Nifti1Image reference.
    """
    if isinstance(ref, tuple) or isinstance(ref, list):
        ref_affine, ref_dimension = ref
        name = 'reference'
    else:
        # Expecting nibabel reference
        ref_affine = ref.affine
        ref_dimension = ref.shape
        name = ref.get_filename()

    affine = hdf5_handle.attrs['affine']
    dimensions = hdf5_handle.attrs['dimensions']
    if not (np.allclose(affine, ref_affine, atol=1e-03)
            and np.array_equal(dimensions, ref_dimension[0:3])):
        raise IOError('HDF5 file does not have a compatible header with'
                      ' {}'.format(name))


def reconstruct_streamlines_from_hdf5(hdf5_group):
    """
    Function to reconstruct streamlines from hdf5, mainly to facilitate
    decomposition into thousands of connections and decrease I/O usage.

    Parameters
    ----------
    hdf5_group: h5py.group
        Handle to the hdf5 group. Ex: hdf5_file[bundle_key].

    Returns
    -------
    streamlines : list of np.ndarray
        List of streamlines.
    """
    if 'data' not in hdf5_group:
        raise ValueError("Expecting data in bundle's group.")

    data = np.array(hdf5_group['data']).flatten()
    offsets = np.array(hdf5_group['offsets'])
    lengths = np.array(hdf5_group['lengths'])

    return reconstruct_streamlines(data, offsets, lengths)


def construct_hdf5_from_sft(hdf5_handle, sfts, groups_keys='streamlines',
                            save_dps=False, save_dpp=False):
    """
    Create a hdf5 from a SFT.

    Parameters
    ----------
    hdf5_handle: h5py.file
        Opened handle to the hdf5 group.
    sfts: StatefulTractogram or list[StatefulTractogram]
    groups_keys: str or list[str]
        The streamlines' hdf5 group name.
    save_dps: bool
        If True, save the DPS keys to hdf5.
    save_dpp: bool
        If True, save the DPP keys to hdf5.
    """
    if isinstance(sfts, StatefulTractogram):
        sfts = [sfts]
    if isinstance(groups_keys, str):
        groups_keys = [groups_keys]

    assert len(sfts) == len(groups_keys)

    # Prepare header
    construct_hdf5_header(hdf5_handle, sfts[0])

    # Prepare streamline groups
    for sft, group_key in zip(sfts, groups_keys):
        sft.to_vox()
        sft.to_corner()
        group = hdf5_handle.create_group(group_key)
        construct_hdf5_group_from_streamlines(
            group, sft.streamlines,
            sft.data_per_streamline if save_dps else None,
            sft.data_per_point if save_dpp else None)


def construct_hdf5_header(hdf5_handle, ref_sft):
    ref_sft.to_vox()
    ref_sft.to_corner()
    hdf5_handle.attrs['affine'] = ref_sft.affine
    hdf5_handle.attrs['dimensions'] = ref_sft.dimensions
    hdf5_handle.attrs['voxel_sizes'] = ref_sft.voxel_sizes
    hdf5_handle.attrs['voxel_order'] = ref_sft.voxel_order


def construct_hdf5_group_from_streamlines(hdf5_group, streamlines,
                                          dps=None, dpp=None):
    """
    Create a hdf5 group from streamlines.

    Parameters
    ----------
    hdf5_group: h5py.group
        Handle to the hdf5 group. Ex: hdf5_file[bundle_key].
    streamlines: ArraySequence
        The streamlines. Expecting streamlines in voxel space, corner origin.
    dps: dict or None
        The data_per_streamline
    dpp: dict or None
        The data_per_point
    """
    hdf5_group.create_dataset('data', data=streamlines.get_data(),
                              dtype=np.float32)
    hdf5_group.create_dataset('offsets', data=streamlines.copy()._offsets,
                              dtype=np.int64)
    hdf5_group.create_dataset('lengths', data=streamlines.copy()._lengths,
                              dtype=np.int32)
    if dps is not None:
        for dps_key, dps_value in dps.items():
            if dps_key not in ['data', 'offsets', 'lengths']:
                hdf5_group.create_dataset(dps_key, data=dps_value,
                                          dtype=np.float32)
            else:
                raise ValueError("Please do not use data_per_streamline keys "
                                 "'data', 'offsets' or 'lengths', this "
                                 "causes unclear management in the hdf5.")

    if dpp is not None:
        raise NotImplementedError(
            "NOT IMPLEMENTED: Cannot save data_per_point in the hdf5 yet.")
