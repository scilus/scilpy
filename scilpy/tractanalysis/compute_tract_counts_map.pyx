# encoding: utf-8
#cython: profile=False

import cython
import numpy as np
cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_tract_counts_map(array_sequence, tuple shape):
    """
    Receives the array_sequence of indices (so you must call uncompress
    before) and returns the number of streamlines passing by each voxel.

    :param array_sequence: nibabel.streamlines.array_sequence.ArraySequence
    :param shape: 3-tuple. The shape of the anat
    """
    count_map = np.zeros(shape, dtype=np.uint32)
    cdef:
        cnp.int64_t[:] lengths = array_sequence._lengths
        cnp.uint32_t nb_streamlines = len(array_sequence._offsets)
        cnp.uint16_t[:, :] indices = array_sequence.data

        cnp.uint32_t[:, :, :] count_map_view = count_map
        cnp.uint64_t[:, :, :] touched_tags_v = np.zeros(shape, dtype=np.uint64)

        cnp.uint64_t idx_streamline = 0, check = 0
        cnp.uint16_t *begin = &indices[0, 0], *it = begin, *end

    # Pseudocode
    # for streamline_indices in streamlines_indices
    #     for voxel_index in streamline_indices
    #         if voxel_index not added *for this streamline*
    #             increment map at voxel_index

    for idx_streamline in range(nb_streamlines):
        # + 1 otherwise the first streamline (index 0) would never work
        check = idx_streamline + 1

        end = it + lengths[idx_streamline] * 3
        while it != end:
            if touched_tags_v[it[0], it[1], it[2]] != check:
                touched_tags_v[it[0], it[1], it[2]] = check
                count_map_view[it[0], it[1], it[2]] += 1
            it += 3

    return count_map
