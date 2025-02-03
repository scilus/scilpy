# -*- coding: utf-8 -*-
import numpy as np

from scilpy.connectivity.connectivity import \
    compute_triu_connectivity_from_labels


def test_compute_triu_connectivity_from_labels():
    labels = np.asarray([[3, 4, 5, 6],
                         [7, 8, 9, 10]])
    labels = labels[:, :, None]
    print("???? shape", labels.shape)

    # vox space, center origin
    # streamline 1 starts at voxel (0, 1) = label 4,
    #              ends at voxel (0, 3) = label 6
    # streamline 2 starts at label 9, ends at label 7
    # streamline 3 too, but not in the center (vox, corner)
    tractogram = [np.asarray([[0.0, 1],
                              [5, 6],
                              [0, 3]]),
                  np.asarray([[1.0, 2],
                              [1, 0]]),
                  np.asarray([[1.1, 2.2],
                              [0.6, 0.4]])]
    output, _, _, _ = compute_triu_connectivity_from_labels(
        tractogram, labels)

    assert np.array_equal(output.shape, [8, 8])  # 8 labels
    expected_out = np.zeros((8, 8))
    expected_out[1, 3] = 1  # One streamline  with label (4, 6) = (2nd, 4th)
    expected_out[4, 6] = 2  # Two streamlines with label (7, 9) = (5nd, 7th)
    assert np.array_equal(output, expected_out)


def test_compute_connectivity_matrices_from_hdf5():
    # ToDo. We will have to create a test hdf5.
    pass


