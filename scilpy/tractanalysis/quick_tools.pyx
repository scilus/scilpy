# encoding: utf-8
#cython: profile=False

import cython
import numpy as np
cimport numpy as cnp


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# def get_intersection_profile_single(atlas_data, strl_indices):
#     """
#     TODO change
#     Get the indices of the voxels traversed by each streamline; then returns
#     an ArraySequence of indices. Yes, of *indices*. ArraySequence.data is
#     always of type float32 and contains points except here: it's of type
#     uint16 and contain indices. You can use this object exactly as you would
#     use a normal ArraySequence.
#
#     :param streamlines: nibabel.streamlines.array_sequence.ArraySequence
#         should be in voxel space, aligned to corner.
#     """
#     cdef:
#         cnp.npy_intp nb_indices = strl_indices.shape[0]
#         cnp.npy_intp at_point = 0
#
#     label_values = np.full((nb_indices, ), -1, dtype=np.int32)
#     indices_values = np.full((nb_indices, ), -1, dtype=np.int32)
#
#     cdef:
#         cnp.npy_intp index_idx = 0
#         cnp.npy_int32[:, :, :] atlas_view = atlas_data
#         cnp.npy_int32[:] data_view = label_values
#         cnp.npy_int32[:] indices_view = indices_values
#
#
#     data_view[at_point] = atlas_view[strl_indices[0][0],
#                                     strl_indices[0][1],
#                                     strl_indices[0][2]]
#     indices_view[at_point] = 0
#
#     at_point += 1
#
#     for index_idx in range(1, nb_indices):
#         next_val = atlas_view[strl_indices[index_idx][0],
#                               strl_indices[index_idx][1],
#                               strl_indices[index_idx][2]]
#
#         if next_val != data_view[at_point - 1]:
#             data_view[at_point] = next_val
#             indices_view[at_point] = index_idx
#             at_point += 1
#
#     label_values.resize((at_point,), refcheck=False)
#     indices_values.resize((at_point,), refcheck=False)
#
#     return (label_values, indices_values)

@cython.boundscheck(False)
@cython.wraparound(False)
def get_next_real_point(points_to_index, vox_index):
    cdef:
        int next_point = -1
        int map_idx = -1
        int nb_points_to_index
        int internal_vox_index
        cnp.npy_ulong[:] pts_to_index_view

    nb_points_to_index = len(points_to_index)
    internal_vox_index = vox_index
    pts_to_index_view = points_to_index

    while map_idx < internal_vox_index and next_point < nb_points_to_index:
        next_point += 1
        #map_idx = points_to_index[next_point]
        map_idx = pts_to_index_view[next_point]

    return next_point


@cython.boundscheck(False)
@cython.wraparound(False)
def get_previous_real_point(points_to_index, vox_index):
    cdef:
        int previous_point
        int map_index
        int nb_points_to_index
        int internal_vox_index
        cnp.npy_ulong[:] pts_to_index_view

    nb_points_to_index = len(points_to_index)
    previous_point = nb_points_to_index
    internal_vox_index = vox_index
    map_idx = internal_vox_index + 1
    pts_to_index_view = points_to_index

    while map_idx > internal_vox_index and previous_point >= 0:
        previous_point -= 1
        map_idx = pts_to_index_view[previous_point]

    return previous_point
