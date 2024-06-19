import itertools
import multiprocessing
import numpy as np

from dipy.io.streamline import load_tractogram
from scilpy.tractanalysis.grid_intersections import grid_intersections


def _compute_fixel_density_parallel(args):
    peaks = args[0]
    max_theta = args[1]
    bundle = args[2]

    sft = load_tractogram(bundle, 'same')
    sft.to_vox()
    sft.to_corner()

    fixel_density_maps = np.zeros((peaks.shape[:-1]) + (5,))

    min_cos_theta = np.cos(np.radians(max_theta))

    all_crossed_indices = grid_intersections(sft.streamlines)
    for crossed_indices in all_crossed_indices:
        segments = crossed_indices[1:] - crossed_indices[:-1]
        seg_lengths = np.linalg.norm(segments, axis=1)

        # Remove points where the segment is zero.
        # This removes numpy warnings of division by zero.
        non_zero_lengths = np.nonzero(seg_lengths)[0]
        segments = segments[non_zero_lengths]
        seg_lengths = seg_lengths[non_zero_lengths]

        # Those starting points are used for the segment vox_idx computations
        seg_start = crossed_indices[non_zero_lengths]
        vox_indices = (seg_start + (0.5 * segments)).astype(int)

        normalized_seg = np.reshape(segments / seg_lengths[..., None], (-1, 3))

        for vox_idx, seg_dir in zip(vox_indices, normalized_seg):
            vox_idx = tuple(vox_idx)
            peaks_at_idx = peaks[vox_idx].reshape((5, 3))

            cos_theta = np.abs(np.dot(seg_dir.reshape((-1, 3)),
                                      peaks_at_idx.T))

            if (cos_theta > min_cos_theta).any():
                lobe_idx = np.argmax(np.squeeze(cos_theta), axis=0)  # (n_segs)
                # TODO Change that for commit weight if given
                fixel_density_maps[vox_idx][lobe_idx] += 1

    return fixel_density_maps


def compute_fixel_density(peaks, bundles, max_theta=45, nbr_processes=None):
    """Compute the fixel density per bundle. Can use parallel processing.

    Parameters
    ----------
    peaks: np.ndarray (x, y, z, 15)
        Five principal fiber orientations for each voxel.
    bundles : list or np.array (N)
        List of (N) paths to bundles.
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

    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(_compute_fixel_density_parallel,
                       zip(itertools.repeat(peaks),
                           itertools.repeat(max_theta),
                           bundles))
    pool.close()
    pool.join()

    fixel_density = np.moveaxis(np.asarray(results), 0, -1)

    return fixel_density
