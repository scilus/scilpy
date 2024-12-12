import itertools
import multiprocessing
import numpy as np

from dipy.io.streamline import load_tractogram
from scilpy.tractanalysis.voxel_boundary_intersection import\
    subdivide_streamlines_at_voxel_faces


def _fixel_density_parallel(args):
    (peaks, max_theta, dps_key, bundle) = args

    return _fixel_density_single_bundle(bundle, peaks, max_theta, dps_key)


def _fixel_density_single_bundle(bundle, peaks, max_theta, dps_key):
    sft = load_tractogram(bundle, 'same')
    sft.to_vox()
    sft.to_corner()

    fixel_density_maps = np.zeros((peaks.shape[:-1]) + (5,))

    min_cos_theta = np.cos(np.radians(max_theta))

    all_split_streamlines =\
        subdivide_streamlines_at_voxel_faces(sft.streamlines)
    for i, split_streamlines in enumerate(all_split_streamlines):
        segments = split_streamlines[1:] - split_streamlines[:-1]
        seg_lengths = np.linalg.norm(segments, axis=1)

        # Remove points where the segment is zero.
        # This removes numpy warnings of division by zero.
        non_zero_lengths = np.nonzero(seg_lengths)[0]
        segments = segments[non_zero_lengths]
        seg_lengths = seg_lengths[non_zero_lengths]

        # Those starting points are used for the segment vox_idx computations
        seg_start = split_streamlines[non_zero_lengths]
        vox_indices = (seg_start + (0.5 * segments)).astype(int)

        normalized_seg = np.reshape(segments / seg_lengths[..., None], (-1, 3))

        weight = 1
        if dps_key:
            weight = sft.data_per_streamline[dps_key][i]

        for vox_idx, seg_dir in zip(vox_indices, normalized_seg):
            vox_idx = tuple(vox_idx)
            peaks_at_idx = peaks[vox_idx].reshape((5, 3))

            cos_theta = np.abs(np.dot(seg_dir.reshape((-1, 3)),
                                      peaks_at_idx.T))

            if (cos_theta > min_cos_theta).any():
                lobe_idx = np.argmax(np.squeeze(cos_theta), axis=0)  # (n_segs)
                fixel_density_maps[vox_idx][lobe_idx] += weight

    return fixel_density_maps


def fixel_density(peaks, bundles, dps_key=None, max_theta=45,
                  nbr_processes=None):
    """Compute the fixel density map per bundle. Can use parallel processing.

    Parameters
    ----------
    peaks : np.ndarray (x, y, z, 15)
        Five principal fiber orientations for each voxel.
    bundles : list or np.array (N)
        List of (N) paths to bundles.
    dps_key : string, optional
        Key to the data_per_streamline to use as weight instead of the number
        of streamlines.
    max_theta : int, optional
        Maximum angle between streamline and peak to be associated.
    nbr_processes : int, optional
        The number of subprocesses to use.
        Default: multiprocessing.cpu_count()

    Returns
    -------
    fixel_density : np.ndarray (x, y, z, 5, N)
        Density per fixel per bundle.
    """
    nbr_processes = multiprocessing.cpu_count() \
        if nbr_processes is None or nbr_processes <= 0 \
        else nbr_processes

    # Separating the case nbr_processes=1 to help get good coverage metrics
    # (codecov does not deal well with multiprocessing)
    if nbr_processes == 1:
        results = []
        for b in bundles:
            results.append(
                _fixel_density_single_bundle(b, peaks, max_theta, dps_key))
    else:
        pool = multiprocessing.Pool(nbr_processes)
        results = pool.map(_fixel_density_parallel,
                           zip(itertools.repeat(peaks),
                               itertools.repeat(max_theta),
                               itertools.repeat(dps_key),
                               bundles))
        pool.close()
        pool.join()

    fixel_density = np.moveaxis(np.asarray(results), 0, -1)

    return fixel_density


def maps_to_masks(maps, abs_thr, rel_thr, norm, nb_bundles):
    """Compute the fixel density masks from fixel density maps.

    Parameters
    ----------
    maps : np.ndarray (x, y, z, 5, N)
        Density per fixel per bundle.
    abs_thr : float
        Value of density maps threshold to obtain density masks, in number of
        streamlines or streamline weighting.
    rel_thr : float
        Value of density maps threshold to obtain density masks, as a ratio of
        the normalized density. Must be between 0 and 1.
    norm : string, ["fixel", "voxel", "none"]
        Way of normalizing the density maps. If fixel, will normalize the maps
        per fixel, in each voxel. If voxel, will normalize the maps per voxel.
        If none, will not normalize the maps.
    nb_bundles : int (N)
        Number of bundles (N).

    Returns
    -------
    masks : np.ndarray (x, y, z, 5, N)
        Density masks per fixel per bundle.
    maps : np.ndarray (x, y, z, 5, N)
        Normalized density maps per fixel per bundle.
    """
    # Apply a threshold on the number of streamlines
    masks_abs = maps > abs_thr

    # Normalizing the density maps per voxel or fixel
    fixel_sum = np.sum(maps, axis=-1)
    voxel_sum = np.sum(fixel_sum, axis=-1)
    for i in range(nb_bundles):
        if norm == "voxel":
            maps[..., 0, i] /= voxel_sum
            maps[..., 1, i] /= voxel_sum
            maps[..., 2, i] /= voxel_sum
            maps[..., 3, i] /= voxel_sum
            maps[..., 4, i] /= voxel_sum
        elif norm == "fixel":
            maps[..., i] /= fixel_sum

    # Apply a threshold on the normalized density
    if norm == "voxel" or norm == "fixel":
        masks_rel = maps > rel_thr
    else:
        masks_rel = maps > 0
    # Compute the fixel density masks from the rel and abs versions
    masks = masks_rel * masks_abs

    return masks.astype(np.uint8), maps
