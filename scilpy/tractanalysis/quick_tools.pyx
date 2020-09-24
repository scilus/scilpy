# encoding: utf-8
#cython: profile=False

import cython
import numpy as np
cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
def get_next_real_point(points_to_index, vox_index):
    cdef:
        int next_point = -1
        int map_idx = -1
        int nb_points_to_index
        int internal_vox_index
        cnp.npy_uint16[:] pts_to_index_view

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
        cnp.npy_uint16[:] pts_to_index_view

    nb_points_to_index = len(points_to_index)
    previous_point = nb_points_to_index
    internal_vox_index = vox_index
    map_idx = internal_vox_index + 1
    pts_to_index_view = points_to_index

    while map_idx > internal_vox_index and previous_point >= 0:
        previous_point -= 1
        map_idx = pts_to_index_view[previous_point]

    return previous_point
