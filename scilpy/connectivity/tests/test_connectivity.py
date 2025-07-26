# -*- coding: utf-8 -*-
import numpy as np

from scilpy.connectivity.connectivity import \
    compute_triu_connectivity_from_labels


def test_compute_triu_connectivity_from_labels():
    labels = np.asarray([[3, 4, 5, 6],
                         [7, 8, 9, 10],
                         [11, 12, 13, 14]])
    labels = labels[:, :, None]
    labels = np.concatenate([labels, labels, labels], axis=-1)
    # Data shape: (3, 4, 3)

    # vox space, center origin
    # streamline 1 : Easy (middle voxel).
    #              starts at voxel (1, 1, 1) = label 8,
    #              ends at voxel (1.1, 0.8, 1.1) = label 8.
    # streamline 2: Border management, center of the voxel
    #              starts at voxel (0, 1, 0) = label 4
    #              ends at voxel (2, 0, 0) = label 11
    # streamline 3: Idem, but not in the center  (vox, center)
    tractogram = [np.asarray([[1.0, 1.0, 1.0],
                              [5.0, 6.1, 10.1],
                              [1.1, 0.8, 1.1]]),
                  np.asarray([[0.0, 1.0, 0.0],
                              [2.0, 0.0, 0.0]]),
                  np.asarray([[-0.3, 0.8, -0.2],
                              [2.1, 0.3, -0.2]])]
    output, _, _, _ = compute_triu_connectivity_from_labels(
        tractogram, labels)

    assert np.array_equal(output.shape, [12, 12])  # 12 labels
    expected_out = np.zeros((12, 12))
    expected_out[5, 5] = 1  # One streamline with label (8, 8) => [5, 5]
    expected_out[1, 8] = 2  # Two streamlines with label (4, 11) = [1, 8]
    assert np.array_equal(output, expected_out)


def test_compute_connectivity_matrices_from_hdf5():
    # ToDo. We will have to create a test hdf5.
    pass
