# -*- coding: utf-8 -*-
import logging

from dipy.core.gradients import get_bval_indices
import numpy as np

from scilpy.gradients.bvec_bval_tools import B0ExtractionStrategy
from scilpy.image.utils import volume_iterator


def extract_dwi_shell(dwi, bvals, bvecs, bvals_to_extract, tol=20,
                      block_size=None):
    """Extracts the DWI volumes that are on specific b-value shells. Many
    shells can be extracted at once by specifying multiple b-values. The
    extracted volumes are in the same order as in the original file.

    If the b-values of a shell are not all identical, use the --tolerance
    argument to adjust the accepted interval. For example, a b-value of 2000
    and a tolerance of 20 will extract all volumes with a b-values from 1980 to
    2020.

    Files that are too large to be loaded in memory can still be processed by
    setting the --block-size argument. A block size of X means that X DWI
    volumes are loaded at a time for processing.

    Parameters
    ----------
    dwi : nib.Nifti1Image
        Original multi-shell volume.
    bvals : ndarray
        The b-values in FSL format.
    bvecs : ndarray
        The b-vectors in FSL format.
    bvals_to_extract : list of int
        The list of b-values to extract.
    tol : int
        The tolerated gap between the b-values to extract and the actual
        b-values.
    block_size : int, optional
        Load the data using this block size. Useful when the data is too
        large to be loaded in memory.

    Returns
    -------
    indices : ndarray
        Indices of the volumes corresponding to the provided b-values.
    shell_data : ndarray
        Volumes corresponding to the provided b-values.
    output_bvals : ndarray
        Selected b-values (raw values, not rounded values, even when tol is
        given).
    output_bvecs : ndarray
        Selected b-vectors.

    """
    indices = [get_bval_indices(bvals, shell, tol=tol)
               for shell in bvals_to_extract]
    indices = np.unique(np.sort(np.hstack(indices)))

    if len(indices) == 0:
        raise ValueError("There are no volumes that have the supplied b-values"
                         ": {}".format(bvals_to_extract))

    logging.info(
        "Extracting shells [{}], with number of images per shell [{}], "
        "from {} images from {}."
        .format(" ".join([str(b) for b in bvals_to_extract]),
                " ".join([str(len(get_bval_indices(bvals, shell, tol=tol)))
                          for shell in bvals_to_extract]),
                len(bvals), dwi.get_filename()))

    if block_size is None:
        block_size = dwi.shape[-1]

    # Load the shells by iterating through blocks of volumes. This approach
    # is slower for small files, but allows very big files to be split
    # with less memory usage.
    shell_data = np.zeros((dwi.shape[:-1] + (len(indices),)))
    for vi, data in volume_iterator(dwi, block_size):
        in_volume = np.array([i in vi for i in indices])
        in_data = np.array([i in indices for i in vi])
        shell_data[..., in_volume] = data[..., in_data]

    output_bvals = bvals[indices].astype(int)
    output_bvecs = bvecs[indices, :]

    return indices, shell_data, output_bvals, output_bvecs


def extract_b0(dwi, b0_mask, extract_in_cluster=False,
               strategy=B0ExtractionStrategy.MEAN, block_size=None):
    """
    Extract a set of b0 volumes from a dwi dataset

    Parameters
    ----------
    dwi : nib.Nifti1Image
        Original multi-shell volume.
    b0_mask: ndarray of bool
        Mask over the time dimension (4th) identifying b0 volumes.
    extract_in_cluster: bool
        Specify to extract b0's in each continuous sets of b0 volumes
        appearing in the input data.
    strategy: enum
        The extraction strategy. Must be one choice available in our enum
        B0ExtractionStrategy: either select the first b0 found, select
        them all or average them. When used in conjunction with the
        extract_in_cluster parameter set to True, the strategy is applied
        individually on each continuous set found.
    block_size : int, optional
        Load the data using this block size. Useful when the data is too
        large to be loaded in memory.

    Returns
    -------
    b0_data : ndarray
        Extracted b0 volumes. 3D with one b0, else 4D.
    """

    indices = np.where(b0_mask)[0]

    if block_size is None:
        block_size = dwi.shape[-1]

    if not extract_in_cluster and strategy == B0ExtractionStrategy.FIRST:
        idx = np.min(indices)
        output_b0 = dwi.dataobj[..., idx:idx + 1].squeeze()
    else:
        # Generate list of clustered b0 in the data
        mask = np.ma.masked_array(b0_mask)
        mask[~b0_mask] = np.ma.masked
        b0_clusters = np.ma.notmasked_contiguous(mask, axis=0)

        if extract_in_cluster or strategy == B0ExtractionStrategy.ALL:
            if strategy == B0ExtractionStrategy.ALL:
                time_d = len(indices)
            else:
                time_d = len(b0_clusters)

            output_b0 = np.zeros(dwi.shape[:-1] + (time_d,))

            for idx, cluster in enumerate(b0_clusters):
                if strategy == B0ExtractionStrategy.FIRST:
                    data = dwi.dataobj[..., cluster.start:cluster.start + 1]
                    output_b0[..., idx] = data.squeeze()
                else:
                    vol_it = volume_iterator(dwi, block_size,
                                             cluster.start, cluster.stop)

                    for vi, data in vol_it:
                        if strategy == B0ExtractionStrategy.ALL:
                            in_volume = np.array([i in vi for i in indices])
                            output_b0[..., in_volume] = data
                        elif strategy == B0ExtractionStrategy.MEAN:
                            output_b0[..., idx] += np.sum(data, -1)

                    if strategy == B0ExtractionStrategy.MEAN:
                        output_b0[..., idx] /= cluster.stop - cluster.start

        else:
            output_b0 = np.zeros(dwi.shape[:-1])
            for cluster in b0_clusters:
                vol_it = volume_iterator(dwi, block_size,
                                         cluster.start, cluster.stop)

                for _, data in vol_it:
                    output_b0 += np.sum(data, -1)

            output_b0 /= len(indices)

    return output_b0
