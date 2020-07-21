# encoding: utf-8
#cython: profile=False

from libc.math cimport ceil, fabs, floor, sqrt
from libc.math cimport fmin as cfmin

import cython
import nibabel as nib
import numpy as np
cimport numpy as cnp


cdef struct Pointers:
    # Incremented when we complete a streamline. Saved at the start of each
    # streamline because we need to start anew if we resize data_out
    cnp.npy_intp *lengths_in
    cnp.npy_intp *lengths_out
    cnp.npy_intp *offsets_in
    cnp.npy_intp *offsets_out

    # const, stop condition
    cnp.npy_intp *lengths_in_end

    # Incremented on each read/written point (each float, in fact)
    float *data_in
    float *data_out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def grid_intersections(streamlines):
    cdef:
        cnp.npy_intp nb_streamlines = len(streamlines._lengths)
        cnp.npy_intp at_point = 0

        # Multiplying by 6 is simply a heuristic to avoiding resizing too many
        # times. In my bundles tests, I had either 0 or 1 resize.
        cnp.npy_intp max_points = (streamlines.get_data().size / 6) * 12

    new_array_sequence = nib.streamlines.array_sequence.ArraySequence()
    new_array_sequence._lengths.resize(nb_streamlines)
    new_array_sequence._offsets.resize(nb_streamlines)
    new_array_sequence._data = np.empty(max_points * 3, np.float32)

    cdef:
        cnp.npy_intp[:] lengths_view_in = streamlines._lengths
        cnp.npy_intp[:] offsets_view_in = streamlines._offsets
        float[:, :] data_view_in = streamlines._data
        cnp.npy_intp[:] lengths_view_out = new_array_sequence._lengths
        cnp.npy_intp[:] offsets_view_out = new_array_sequence._offsets
        cnp.float32_t[:] data_view_out = new_array_sequence._data

    cdef Pointers pointers
    pointers.lengths_in = &lengths_view_in[0]
    pointers.lengths_in_end = pointers.lengths_in + nb_streamlines
    pointers.offsets_in = &offsets_view_in[0]
    pointers.data_in = &data_view_in[0, 0]
    pointers.lengths_out = &lengths_view_out[0]
    pointers.offsets_out = &offsets_view_out[0]
    pointers.data_out = &data_view_out[0]

    while 1:
        at_point = _grid_intersections(&pointers, at_point, max_points - 1)
        if pointers.lengths_in == pointers.lengths_in_end:
            # Job finished, we can return the streamlines
            break

        # Resize and point the memoryview and pointer on the right data
        max_points += max_points / 3  # Make it one third bigger
        new_array_sequence._data.resize(max_points * 3, refcheck=False)
        data_view_out = new_array_sequence._data
        pointers.data_out = &data_view_out[0] + at_point * 3

    new_array_sequence._data.resize((at_point, 3), refcheck=False)
    return new_array_sequence


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double norm(double x, double y, double z) nogil:
    cdef double val = sqrt(x*x + y*y + z*z)
    return val


# Changing this to a memview was slower.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void c_get_closest_edge(double *p,
                                    double *direction,
                                    double *edge,
                                    double eps=1.0) nogil:
    edge[0] = floor(p[0] + eps) if direction[0] >= 0.0 else ceil(p[0] - eps)
    edge[1] = floor(p[1] + eps) if direction[1] >= 0.0 else ceil(p[1] - eps)
    edge[2] = floor(p[2] + eps) if direction[2] >= 0.0 else ceil(p[2] - eps)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void copypoint_f(float * a, float * b) nogil:
    for i in range(3):
        b[i] = a[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void copypoint_d(double * a, double * b) nogil:
    for i in range(3):
        b[i] = a[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void copypoint_f2d(float * a, double * b) nogil:
    for i in range(3):
        b[i] = <double>(a[i])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void copypoint_d2f(double * a, float * b) nogil:
    for i in range(3):
        b[i] = <float>(a[i])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cnp.npy_intp _grid_intersections(
        Pointers* pointers,
        cnp.npy_intp at_point,
        cnp.npy_intp max_points) nogil:
    cdef:
        float *backup_data_in
        cnp.npy_intp backup_at_point

        cnp.npy_intp nb_points_in, point_idx, nb_points_out
        double *current_pt = [0.0, 0.0, 0.0]
        double *next_pt = [0.0, 0.0, 0.0]
        double *direction = [0.0, 0.0, 0.0]
        double *current_edge = [0.0, 0.0, 0.0]
        double *intersected_edge = [0.0, 0.0, 0.0]
        double *jittered_point = [-1.0, -1.0, -1.0]

        double direction_norm, remaining_distance
        double length_ratio

    # For each streamline
    while pointers.lengths_in != pointers.lengths_in_end:
        # We must finish the streamline or start again from the start
        backup_data_in = pointers.data_in
        backup_at_point = at_point

        nb_points_out = 0
        nb_points_in = pointers.lengths_in[0]

        jittered_point[0] = -1.0
        jittered_point[1] = -1.0
        jittered_point[2] = -1.0

        # Check the very first point
        copypoint_f(pointers.data_in, pointers.data_out)
        pointers.data_out += 3
        nb_points_out += 1
        at_point += 1

        # Make sure we don't already hit the max.
        if at_point == max_points:
            pointers.data_in = backup_data_in
            return backup_at_point

        for point_idx in range(nb_points_in - 1):
            # Only checking the first element since all three should either be
            # set to -1 when no jittering has taken place, or be > 0 if
            # jittering has taken place, since we are in voxel coordinates.
            if jittered_point[0] > -1.0:
                # Make sure to use the correct point if we needed to jitter
                # it because of an edge case
                copypoint_d(jittered_point, current_pt)
                copypoint_d(jittered_point, current_edge)
                jittered_point[0] = -1.0
                jittered_point[1] = -1.0
                jittered_point[2] = -1.0
            else:
                copypoint_f2d(pointers.data_in, current_pt)
                copypoint_d(current_pt, current_edge)

            # Always advance to next point
            pointers.data_in += 3
            copypoint_f2d(pointers.data_in, next_pt)
            direction[0] = next_pt[0] - current_pt[0]
            direction[1] = next_pt[1] - current_pt[1]
            direction[2] = next_pt[2] - current_pt[2]

            direction_norm = norm(direction[0], direction[1], direction[2])

            # Make sure that the next point is not exactly on a voxel
            # intersection or on the face of the voxel, since the behavior
            # is not easy to define in this case.
            if fabs(next_pt[0] - floor(next_pt[0])) < 1e-8 or\
                fabs(next_pt[1] - floor(next_pt[1])) < 1e-8 or\
                fabs(next_pt[2] - floor(next_pt[2])) < 1e-8:
                # TODO in future, jitter edge or add thickness to deal with
                # case where on an edge / face and the corresponding
                # component of the direction is 0
                next_pt[0] = next_pt[0] - 0.000001 * direction[0]
                next_pt[1] = next_pt[1] - 0.000001 * direction[1]
                next_pt[2] = next_pt[2] - 0.000001 * direction[2]

                # Make sure we don't "underflow" the grid
                if next_pt[0] < 0.0 or next_pt[1] < 0.0 or next_pt[2] < 0.0:
                    next_pt[0] = next_pt[0] + 0.000002 * direction[0]
                    next_pt[1] = next_pt[1] + 0.000002 * direction[1]
                    next_pt[2] = next_pt[2] + 0.000002 * direction[2]

                # Keep it in mind to correctly set when going back in the loop
                copypoint_d(next_pt, jittered_point)

                # Update those
                direction[0] = next_pt[0] - current_pt[0]
                direction[1] = next_pt[1] - current_pt[1]
                direction[2] = next_pt[2] - current_pt[2]

                direction_norm = norm(direction[0], direction[1], direction[2])

            # Set the "remaining_distance" var to compute remaining length of
            # vector to process
            remaining_distance = direction_norm

            # If consecutive coordinates are the same, skip one.
            if direction_norm == 0:
                continue

            while True:
                c_get_closest_edge(current_pt, direction, current_edge)

                # Compute the smallest ratio of direction's length to get to an
                # edge. This effectively means we find the first edge
                # encountered. Set large value for length_ratio.
                length_ratio = 10000
                for dim_idx in range(3):
                    if direction[dim_idx] != 0:
                        length_ratio = cfmin(
                            fabs((current_edge[dim_idx] - current_pt[dim_idx])
                                / direction[dim_idx]),
                            length_ratio)

                # Check if last point is already on an edge
                remaining_distance -= length_ratio * direction_norm
                if remaining_distance < 0.:
                    # Add point. This is needed in this version to make sure
                    # we get the correct segments orientations when the
                    # segment changes orientation between 2 points
                    copypoint_d2f(next_pt, pointers.data_out)
                    pointers.data_out += 3
                    nb_points_out += 1

                    # Are we full yet?
                    at_point += 1
                    if at_point == max_points:
                        pointers.data_in = backup_data_in
                        return backup_at_point
                    break

                # Compute intersected edge
                intersected_edge[0] = current_pt[0] + length_ratio * direction[0]
                intersected_edge[1] = current_pt[1] + length_ratio * direction[1]
                intersected_edge[2] = current_pt[2] + length_ratio * direction[2]

                if intersected_edge[0] != pointers.data_out[-3] or\
                   intersected_edge[1] != pointers.data_out[-2] or\
                   intersected_edge[2] != pointers.data_out[-1]:
                    copypoint_d2f(intersected_edge, pointers.data_out)
                    pointers.data_out += 3
                    nb_points_out += 1

                    # Are we full yet?
                    at_point += 1
                    if at_point == max_points:
                        pointers.data_in = backup_data_in
                        return backup_at_point

                # Advance to next voxel
                move_ratio = length_ratio + 0.00000001
                current_pt[0] = current_pt[0] + move_ratio * direction[0]
                current_pt[1] = current_pt[1] + move_ratio * direction[1]
                current_pt[2] = current_pt[2] + move_ratio * direction[2]

        # Streamline finished, advance
        pointers.data_in += 3
        pointers.lengths_out[0] = nb_points_out
        if backup_at_point == 0:  # If first
            pointers.offsets_out[0] = 0
        else:
            # C negative indexing is supposed to be legal! From C99 §6.5.2.1/2
            pointers.offsets_out[0] =\
                pointers.offsets_out[-1] + pointers.lengths_out[-1]
        pointers.offsets_out += 1
        pointers.offsets_in += 1
        pointers.lengths_in += 1
        pointers.lengths_out += 1

    # At this return, everything is finished
    return at_point
