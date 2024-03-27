# -*- coding: utf-8 -*-

import logging

from dipy.segment.mask import bounding_box
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans

from scilpy.utils.spatial import (get_axis_index,
                                  RAS_AXES_NAMES,
                                  voxel_to_world,
                                  WorldBoundingBox)


def volume_iterator(img, blocksize=1, start=0, end=0):
    """Generator that iterates on volumes of data.

    Parameters
    ----------
    img : nib.Nifti1Image
        Image of a 4D volume with shape X,Y,Z,N
    blocksize : int, optional
        Number of volumes to return in a single batch
    start : int, optional
        Starting iteration index in the 4D volume
    end : int, optional
        Stopping iteration index in the 4D volume
        (the volume at this index is excluded)

    Yields
    -------
    tuple of (list of int, ndarray)
        The ids of the selected volumes, and the selected data as a 4D array
    """
    assert end <= img.shape[-1], "End limit provided is greater than the " \
                                 "total number of volumes in image"

    nb_volumes = img.shape[-1]
    end = end if end else img.shape[-1]

    if blocksize == nb_volumes:
        yield list(range(start, end)), \
              img.get_fdata(dtype=np.float32)[..., start:end]
    else:
        stop = start
        for i in range(start, end - blocksize, blocksize):
            start, stop = i, i + blocksize
            logging.info("Loading volumes {} to {}.".format(start, stop - 1))
            yield list(range(start, stop)), img.dataobj[..., start:stop]

        if stop < end:
            logging.info(
                "Loading volumes {} to {}.".format(stop, end - 1))
            yield list(range(stop, end)), img.dataobj[..., stop:end]


def extract_affine(input_files):
    """Extract the affine from a list of nifti files.

    Parameters
    ----------
    input_files : list of strings (file paths)
        Diffusion data files.

    Returns
    -------
    affine : np.ndarray
        Affine of the nifti volume.
    """
    for input_file in input_files:
        if input_file:
            vol = nib.load(input_file)
            return vol.affine


def check_slice_indices(vol_img, axis_name, slice_ids):
    """Check that the given volume can be sliced at the given slice indices
    along the requested axis.

    Parameters
    ----------
    vol_img : nib.Nifti1Image
        Volume image.
    axis_name : str
        Slicing axis name.
    slice_ids : array-like
        Slice indices.
    """

    if axis_name not in RAS_AXES_NAMES:
        raise NotImplementedError(
            f"Unsupported axis name:\n"
            f"Found: {axis_name}; Available: {RAS_AXES_NAMES.join(', ')}")

    shape = vol_img.shape
    idx = get_axis_index(axis_name)
    _slice_ids = list(filter(lambda x: x > shape[idx], slice_ids))
    if _slice_ids:
        raise ValueError(
            "Slice indices exceed the volume shape along the given axis:\n"
            f"Slices {_slice_ids} exceed shape {shape} along dimension {idx}.")


def split_mask_blobs_kmeans(data, nb_clusters):
    """
    Split a mask between head and tail with k means.

    Parameters
    ----------
    data: numpy.ndarray
        Mask to be split.
    nb_clusters: int
        Number of clusters to split.

    Returns
    -------
    masks: List[np.ndarray]
        The masks for each cluster.
    """

    X = np.argwhere(data)
    k_means = KMeans(n_clusters=nb_clusters).fit(X)

    masks = []
    for i in range(nb_clusters):
        mask_i = np.zeros(data.shape)
        mask_i[tuple(X[np.where(k_means.labels_ == i)].T)] = 1
        masks.append(mask_i)

    return masks


def compute_nifti_bounding_box(img):
    """
    Finds bounding box from data and transforms it in world space for use
    on data with different attributes like voxel size.

    Parameters
    ----------
    img: nib.Nifti1Image
        Input image file.

    Returns
    -------
    wbbox: WorldBoundingBox Object
        Bounding box in world space.
    """
    data = img.get_fdata(dtype=np.float32, caching='unchanged')
    affine = img.affine
    voxel_size = img.header.get_zooms()[0:3]

    voxel_bb_mins, voxel_bb_maxs = bounding_box(data)

    world_bb_mins = voxel_to_world(voxel_bb_mins, affine)
    world_bb_maxs = voxel_to_world(voxel_bb_maxs, affine)
    wbbox = WorldBoundingBox(world_bb_mins, world_bb_maxs, voxel_size)

    return wbbox
