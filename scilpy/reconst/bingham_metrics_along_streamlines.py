# -*- coding: utf-8 -*-

from dipy.io.stateful_tractogram import StatefulTractogram
import numpy as np
from scilpy.reconst.bingham import NB_PARAMS, bingham_from_array


def fiber_density_map_along_streamlines(sft: StatefulTractogram,
                                        bingham_coeffs, fiber_density):
    """
    """
    volume_shape = bingham_coeffs.shape[:-1]
    n_coeffs = bingham_coeffs.shape[-1]
    max_lobes = n_coeffs // NB_PARAMS

    # convert bingham coefficients to peak direction
    def _peaks_from_bingham_coeffs(coeffs):
        dirs = np.array([bingham_from_array(coeffs[i:i+NB_PARAMS])
                         .peak_direction() for i in range(max_lobes)])
        dirs = np.squeeze(dirs)
        return dirs

    # create a peaks volume from bingham coefficients volume
    bingham_peaks = np.array([_peaks_from_bingham_coeffs(coeffs)
                              for coeffs
                              in bingham_coeffs.reshape((-1, n_coeffs))])
    bingham_peaks = bingham_peaks.reshape(volume_shape +
                                          bingham_peaks.shape[-2:])

    # safety for having streamlines origin in corner of voxel space
    sft.to_vox()
    sft.to_corner()

    bundle_mean_fd = np.zeros(volume_shape)
    for streamline in sft.streamlines:
        indices = np.asarray(streamline.astype(int)).T
        indices_1d = np.ravel_multi_index(indices, volume_shape)
        print(indices.shape)
        print(len(indices_1d))
        peaks = bingham_peaks[np.unravel_index(indices_1d, volume_shape)]
        print(peaks.shape)
        # dot product between streamline segments and
        # peaks in corresponding voxels
        1/0

    return bingham_coeffs
