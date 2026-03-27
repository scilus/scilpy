# -*- coding: utf-8 -*-
import os

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map

from scilpy.tractanalysis.afd_along_streamlines import \
    afd_map_along_streamlines
from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict


fetch_data(get_testing_files_dict(), keys=['processing.zip'])


def test_afd_map_along_streamlines():
    fodf_img = nib.load(os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz'))
    fodf = fodf_img.get_fdata(dtype=np.float32)
    fodf_mask = np.sum(fodf, axis=-1) > 0

    peaks = nib.load(os.path.join(SCILPY_HOME, 'processing', 'peaks.nii.gz'))
    peaks = peaks.get_fdata(dtype=np.float32)
    peaks = peaks[:, :, :, 0:3]  # Keeping only the first peak

    # Creating a few very short streamlines (2 points) aligned on peaks
    # in the middle of the image (shape 57x67x56)
    streamlines = []
    for i in range(23, 26):
        start_pt = np.asarray([i + 0.1, i + 0.1, i + 0.1])
        peak = peaks[tuple(start_pt.astype(int))]
        peak = np.asarray(peak) / np.linalg.norm(peak)
        second_pt = np.asarray(start_pt) + peak
        streamlines.append(np.vstack([start_pt, second_pt], dtype=np.float32))
    sft = StatefulTractogram(streamlines=streamlines,
                             reference=fodf_img,
                             space=Space.VOX, origin=Origin('corner'))

    # Processing
    afd_map, rd_map = afd_map_along_streamlines(
        sft, fodf_img, 'descoteaux07',
        length_weighting=False, is_legacy=False)

    # Should have the same size as fodf
    assert np.array_equal(fodf_img.shape[0:3], afd_map.shape)

    # Should only have values where we have streamlines + inside fodf mask.
    # Note. If the streamline starts exactly on the border of the voxel,
    # we get weird results, because the code for afd uses segments, which means
    # that the coordinate [i, i, i] may belong to the voxel [i, i, i] or to its
    # neighbor, depending on the direction of the segment, but the density_map
    # always considers it to be in voxel [i, i, i]. But ok, streamlines exactly
    # on a border should not happen too often in real life.
    density_map = compute_tract_counts_map(sft.streamlines, sft.dimensions)
    density_mask = density_map > 0
    mask = np.logical_and(density_mask, fodf_mask)
    assert np.array_equal(mask, afd_map != 0)

    # Now processing again, but with more of the same streamlines
    # AFD will be the same (divided by weight map = by number of streamlines)
    streamlines2 = [s for s in streamlines]
    for translation in [0.1, 0.11, 0.12]:
        streamlines2 += [s + translation for s in streamlines]
    sft.streamlines = streamlines2
    afd_map2, rd_map2 = afd_map_along_streamlines(
        sft, fodf_img, 'descoteaux07',
        length_weighting=False, is_legacy=False)
    assert np.count_nonzero(afd_map) == np.count_nonzero(afd_map2)
    assert np.count_nonzero(rd_map) == np.count_nonzero(rd_map2)
    assert np.allclose(afd_map, afd_map2)
    assert np.allclose(rd_map, rd_map2)

    # Now processing again, but with rotated peaks. Values should be in the
    # same voxels, but different
    streamlines2 = [s for s in streamlines]
    for s in streamlines:
        # Same starting point. Moving a little the end point, but we want to
        # stay in the same voxel
        modified_s = [s[0, :], s[1, :] + 0.1]
        streamlines2.append(modified_s)
    sft.streamlines = streamlines2
    afd_map2, rd_map2 = afd_map_along_streamlines(
        sft, fodf_img, 'descoteaux07',
        length_weighting=False, is_legacy=False)
    assert np.count_nonzero(afd_map) == np.count_nonzero(afd_map2)
    assert np.count_nonzero(rd_map) == np.count_nonzero(rd_map2)
    assert not np.allclose(afd_map, afd_map2)
    assert not np.allclose(rd_map, rd_map2)
