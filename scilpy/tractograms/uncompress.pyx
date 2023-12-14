# encoding: utf-8
#cython: profile=False
#cython: language_level=3

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
    cnp.uint16_t *data_out

    # To bookkeep the final index related to a streamline point
    cnp.npy_intp *pti_lengths_out
    cnp.npy_intp *pti_offsets_out
    cnp.uint16_t *points_to_index_out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def uncompress(streamlines, return_mapping=False):
    """
    Get the indices of the voxels traversed by each streamline; then returns
    an ArraySequence of indices, i.e. [i, j, k] coordinates.

    Parameters
    ----------
    streamlines: nibabel.streamlines.array_sequence.ArraySequence
        Should be in voxel space, aligned to corner.
    return_mapping: bool
        If true, also returns the points_to_idx.

    Returns
    -------
    indices: nibabel.streamlines.array_sequence.ArraySequence
        An array of length nb_streamlines. Each element is a np.ndarray of
        shape (nb_voxels, 3) containing indices. All 3D coordinates are unique.
        Note. ArraySequence.get_data() is always of type float32 and contains
        points except here: it's of type uint16 and contain indices. You can use
        this object exactly as you would use a normal ArraySequence.
    points_to_idx: nibabel.streamlines.array_sequence.ArraySequence (optional)
        An array of length nb_streamlines. Each element is a np.ndarray of
        shape (nb_points) containing, for each streamline point, the associated
        voxel in indices.
        Note: Some points are associated to the same index, if they were in the
        same voxel. Some voxels are associated to no point, if they were
        traversed by a segment but contained no point.
    """
    cdef:
        cnp.npy_intp nb_streamlines = len(streamlines._lengths)
        cnp.npy_intp at_point = 0

        # Multiplying by 6 is simply a heuristic to avoiding resizing too many
        # times. In my bundles tests, I had either 0 or 1 resize.
        cnp.npy_intp max_points = (streamlines._data.size / 3)

    new_array_sequence = nib.streamlines.array_sequence.ArraySequence()
    new_array_sequence._lengths.resize(nb_streamlines)
    new_array_sequence._offsets.resize(nb_streamlines)
    new_array_sequence._data = np.empty(max_points * 3, np.uint16)

    points_to_index = nib.streamlines.array_sequence.ArraySequence()
    points_to_index._lengths.resize(nb_streamlines)
    points_to_index._offsets.resize(nb_streamlines)
    points_to_index._data = np.zeros(int(streamlines._data.size / 3), np.uint16)

    cdef:
        cnp.npy_intp[:] lengths_view_in = streamlines._lengths
        cnp.npy_intp[:] offsets_view_in = streamlines._offsets
        float[:, :] data_view_in = streamlines._data
        cnp.npy_intp[:] lengths_view_out = new_array_sequence._lengths
        cnp.npy_intp[:] offsets_view_out = new_array_sequence._offsets
        cnp.uint16_t[:] data_view_out = new_array_sequence._data
        cnp.npy_intp[:] pti_lengths_view_out = points_to_index._lengths
        cnp.npy_intp[:] pti_offsets_view_out = points_to_index._offsets
        cnp.uint16_t[:] points_to_index_view_out = points_to_index._data

    cdef Pointers pointers
    pointers.lengths_in = &lengths_view_in[0]
    pointers.lengths_in_end = pointers.lengths_in + nb_streamlines
    pointers.offsets_in = &offsets_view_in[0]
    pointers.data_in = &data_view_in[0, 0]
    pointers.lengths_out = &lengths_view_out[0]
    pointers.offsets_out = &offsets_view_out[0]
    pointers.data_out = &data_view_out[0]
    pointers.pti_lengths_out = &pti_lengths_view_out[0]
    pointers.pti_offsets_out = &pti_offsets_view_out[0]
    pointers.points_to_index_out = &points_to_index_view_out[0]

    while 1:
        at_point = _uncompress(&pointers, at_point, max_points - 1)
        if pointers.lengths_in == pointers.lengths_in_end:
            # Job finished, we can return the streamlines
            break

        # Resize and point the memoryview and pointer on the right data
        max_points += max_points / 3  # Make it one third bigger
        new_array_sequence._data.resize(max_points * 3, refcheck=False)
        data_view_out = new_array_sequence._data
        pointers.data_out = &data_view_out[0] + at_point * 3

        if at_point == 0:
            pointers.points_to_index_out = &points_to_index_view_out[0]
        else:
            pointers.points_to_index_out = &points_to_index_view_out[0] + pointers.offsets_in[0]

    new_array_sequence._data.resize((at_point, 3), refcheck=False)

    if not return_mapping:
        return new_array_sequence
    else:
        return (new_array_sequence, points_to_index)


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
@cython.cdivision(True)
cdef cnp.npy_intp _uncompress(
        Pointers* pointers,
        cnp.npy_intp at_point,
        cnp.npy_intp max_points) nogil:
    cdef:
        float *backup_data_in
        cnp.npy_intp backup_at_point

        cnp.npy_intp nb_points_in, point_idx, nb_points_out, prev_index, \
                     nb_points_out_pti
        double *current_pt = [0.0, 0.0, 0.0]
        double *next_pt = [0.0, 0.0, 0.0]
        double *direction = [0.0, 0.0, 0.0]
        double *current_edge = [0.0, 0.0, 0.0]
        double *jittered_point = [-1.0, -1.0, -1.0]

        double direction_norm, remaining_distance
        double length_ratio, move_ratio
        cnp.uint16_t x = 0, y = 0, z = 0
        cnp.uint16_t last_x, last_y, last_z

    # For each streamline
    while pointers.lengths_in != pointers.lengths_in_end:
        # We must finish the streamline or start again from the beginning
        backup_data_in = pointers.data_in
        backup_at_point = at_point

        nb_points_out = 0
        nb_points_out_pti = 0
        nb_points_in = pointers.lengths_in[0]

        jittered_point[0] = -1.0
        jittered_point[1] = -1.0
        jittered_point[2] = -1.0

        if backup_at_point == 0:  # If first
            prev_index = 0
        else:
            # C negative indexing is supposed to be legal! From C99 §6.5.2.1/2
            prev_index = pointers.offsets_out[-1] + pointers.lengths_out[-1]

        # Check the very first point
        x = <cnp.uint16_t>(pointers.data_in[0])
        y = <cnp.uint16_t>(pointers.data_in[1])
        z = <cnp.uint16_t>(pointers.data_in[2])
        pointers.data_out[0] = x
        pointers.data_out[1] = y
        pointers.data_out[2] = z
        pointers.points_to_index_out[0] = 0
        pointers.points_to_index_out += 1
        pointers.data_out += 3
        nb_points_out += 1
        nb_points_out_pti += 1
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
                current_edge[0] = current_pt[0] = jittered_point[0]
                current_edge[1] = current_pt[1] = jittered_point[1]
                current_edge[2] = current_pt[2] = jittered_point[2]
                jittered_point[0] = -1.0
                jittered_point[1] = -1.0
                jittered_point[2] = -1.0
            else:
                current_edge[0] = current_pt[0] = pointers.data_in[0]
                current_edge[1] = current_pt[1] = pointers.data_in[1]
                current_edge[2] = current_pt[2] = pointers.data_in[2]

            # Always advance to next point
            pointers.data_in += 3
            next_pt[0] = pointers.data_in[0]
            next_pt[1] = pointers.data_in[1]
            next_pt[2] = pointers.data_in[2]
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
                jittered_point[0] = next_pt[0]
                jittered_point[1] = next_pt[1]
                jittered_point[2] = next_pt[2]

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
                    pointers.points_to_index_out[0] = nb_points_out - 1
                    pointers.points_to_index_out += 1
                    nb_points_out_pti += 1
                    break

                # Find the coordinates of voxel containing current point, to
                # tag it in the map
                move_ratio = length_ratio + 0.00000001
                current_pt[0] = current_pt[0] + move_ratio * direction[0]
                current_pt[1] = current_pt[1] + move_ratio * direction[1]
                current_pt[2] = current_pt[2] + move_ratio * direction[2]

                x = <cnp.uint16_t>(current_pt[0])
                y = <cnp.uint16_t>(current_pt[1])
                z = <cnp.uint16_t>(current_pt[2])
                pointers.data_out[0] = x
                pointers.data_out[1] = y
                pointers.data_out[2] = z
                pointers.data_out += 3
                nb_points_out += 1

                # Are we full yet? We return 1 before the max because we would
                # still need to add the last point.
                at_point += 1
                if at_point == max_points:
                    pointers.data_in = backup_data_in
                    return backup_at_point

        # We ignore the last point in the loop above, so we must increment it
        pointers.data_in += 3

        # Check last point
        last_x = <cnp.uint16_t>next_pt[0]
        last_y = <cnp.uint16_t>next_pt[1]
        last_z = <cnp.uint16_t>next_pt[2]
        if x != last_x or y != last_y or z != last_z:
            with gil:
                print('Got inside last')

            pointers.data_out[0] = last_x
            pointers.data_out[1] = last_y
            pointers.data_out[2] = last_z
            pointers.data_out += 3
            pointers.points_to_index_out[0] = nb_points_out
            pointers.points_to_index_out += 1
            nb_points_out += 1
            nb_points_out_pti += 1
            at_point += 1

        # Streamline finished, advance
        pointers.lengths_out[0] = nb_points_out
        pointers.pti_lengths_out[0] = nb_points_out_pti
        if backup_at_point == 0:  # If first
            pointers.offsets_out[0] = 0
            pointers.pti_offsets_out[0] = 0
        else:
            # C negative indexing is supposed to be legal! From C99 §6.5.2.1/2
            pointers.offsets_out[0] =\
                pointers.offsets_out[-1] + pointers.lengths_out[-1]
            pointers.pti_offsets_out[0] =\
                pointers.pti_offsets_out[-1] + pointers.pti_lengths_out[-1]
        pointers.offsets_out += 1
        pointers.offsets_in += 1
        pointers.lengths_in += 1
        pointers.lengths_out += 1
        pointers.pti_offsets_out += 1
        pointers.pti_lengths_out += 1

    # At this return, everything is finished
    return at_point
