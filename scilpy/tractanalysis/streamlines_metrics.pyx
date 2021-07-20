# encoding: utf-8
#cython: profile=False

# http://www.cse.yorku.ca/~amana/research/grid.pdf
# http://www.flipcode.com/archives/Raytracing_Topics_Techniques-Part_4_Spatial_Subdivisions.shtml


cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport sqrt, floor, ceil, fabs
from libc.math cimport fmin as cfmin

# Changing this to a memview was slower.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double norm(double x, double y, double z) nogil:
    cdef double val = sqrt(x*x + y*y + z*z)
    return val


# Changing this to a memview was slower.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void c_get_closest_edge(double p_x, double p_y, double p_z,
                                    double d_x, double d_y, double d_z,
                                    np.double_t[:] edge,
                                    double eps=1.) nogil:
     edge[0] = floor(p_x + eps) if d_x >= 0.0 else ceil(p_x - eps)
     edge[1] = floor(p_y + eps) if d_y >= 0.0 else ceil(p_y - eps)
     edge[2] = floor(p_z + eps) if d_z >= 0.0 else ceil(p_z - eps)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# IMPORTANT: Streamlines should be in voxel space, aligned to corner.
def compute_tract_counts_map(streamlines, vol_dims):
    flags = np.seterr(divide="ignore", under="ignore")

    # Inspired from Dipy track_counts
    vol_dims = np.asarray(vol_dims).astype(int)
    n_voxels = np.prod(vol_dims)

    # This array counts the number of different tracks going through each voxel.
    # Need to keep both the array and the memview on it to be able to
    # reshape and return in the end.
    traversal_tags = np.zeros((n_voxels,), dtype=int)
    cdef np.int_t[:] traversal_tags_v = traversal_tags

    # This array keeps track of whether the current track has already been
    # flagged in a specific voxel.
    cdef np.int_t[:] touched_tags_v = np.zeros((n_voxels,), dtype=int)

    cdef int streamlines_len = len(streamlines)

    if streamlines_len == 0:
        np.seterr(**flags)
        return traversal_tags.reshape(vol_dims)

    # Memview to a streamline instance, which is a numpy array.
    cdef np.double_t[:,:] t = streamlines[0].astype(np.double)

    # Memviews for points and direction vectors.
    cdef np.double_t[:] in_pt = np.zeros(3, dtype=np.double)
    cdef np.double_t[:] next_pt = np.zeros(3, dtype=np.double)
    cdef np.double_t[:] dir_vect = np.zeros(3, dtype=np.double)

    # Memview for the current edge
    cdef np.double_t[:] cur_edge = np.zeros(3, dtype=np.double)

    # Memview for the coordinates of the current voxel
    cdef np.int_t[:] cur_voxel_coords = np.zeros(3, dtype=int)

    # various temporary loop and working variables
    #cdef:
    #    int track_idx   # Index of the track when iterating.
    cdef int tno, pno, cno
    cdef np.npy_intp el_no, v

    cdef int vd[3]
    cdef double vxs[3]
    for cno in range(3):
        vd[cno] = vol_dims[cno]
    # x slice size (C array ordering)
    cdef np.npy_intp x_slice_size = vd[1] * vd[2]

    cdef np.double_t dir_vect_norm, remaining_dist, length_ratio

    for track_idx in range(streamlines_len):
        t = streamlines[track_idx].astype(np.double)

        # This loop is time-critical
        # Changed to -1 because we get the next point in the loop
        for pno in range(t.shape[0] - 1):
            # Assign current and next point, find vector between both,
            # and use the current point as nearest edge for testing.
            for cno in range(3):
                in_pt[cno] = t[pno, cno]
                next_pt[cno] = t[pno + 1, cno]
                dir_vect[cno] = next_pt[cno] - in_pt[cno]
                cur_edge[cno] = in_pt[cno]

            # Compute norm
            dir_vect_norm = norm(dir_vect[0], dir_vect[1], dir_vect[2])

            # If consecutive coordinates are the same, skip one.
            if dir_vect_norm == 0:
                continue

            # Set the "dist" var to compute remaining length of vector to process
            remaining_dist = dir_vect_norm

            # Check if it's already a real edge. If not, find the closest edge.
            # Reverted the condition to help with code prediction
            if floor(cur_edge[0]) != cur_edge[0] and \
               floor(cur_edge[1]) != cur_edge[1] and \
               floor(cur_edge[2]) != cur_edge[2]:
                # All coordinates are not "integers", and therefore, not on the
                # edge. Fetch the closest edge.
                c_get_closest_edge(in_pt[0], in_pt[1], in_pt[2],
                                   dir_vect[0], dir_vect[1], dir_vect[2],
                                   cur_edge)

            # TODO Could condition be optimized?
            while True:
                # Compute the smallest ratio of dir_vect's length to get to an
                # edge. This effectively means we find the first edge
                # encountered
                # Set large value for length_ratio
                length_ratio = 10000
                for cno in range(3):
                    # To avoid dividing by zero.
                    # Gain in performance, since we can use
                    # @cython.cdivision(True)
                    if dir_vect[cno] != 0:
                        length_ratio = cfmin(fabs((cur_edge[cno] - in_pt[cno]) /
                                             dir_vect[cno]), length_ratio)

                remaining_dist -= length_ratio * dir_vect_norm

                # Check if last point is already on an edge
                if remaining_dist < 0 and not fabs(remaining_dist) < 1e-8:
                    break

                # Find the coordinates of voxel containing current point, to
                # tag it in the map
                for cno in range(3):
                    cur_voxel_coords[cno] = <int>floor(in_pt[cno] +
                                                       0.5 * length_ratio *
                                                       dir_vect[cno])

                el_no = cur_voxel_coords[0] * x_slice_size + \
                        cur_voxel_coords[1] * vd[2] + cur_voxel_coords[2]

                # Use + 1 since the first track would be ignored
                if touched_tags_v[el_no] != track_idx + 1:
                    touched_tags_v[el_no] = track_idx + 1
                    traversal_tags_v[el_no] += 1

                # NOTE: in_pt is moved to the closest edge
                for cno in range(3):
                    in_pt[cno] = length_ratio * dir_vect[cno] + in_pt[cno]

                    # Snap really small values to 0.
                    if fabs(in_pt[cno]) <= 1e-16:
                        in_pt[cno] = 0.0

                c_get_closest_edge(in_pt[0], in_pt[1], in_pt[2],
                                   dir_vect[0], dir_vect[1], dir_vect[2],
                                   cur_edge)

        # Add last point
        for cno in range(3):
            cur_voxel_coords[cno] = <int>floor(in_pt[cno] +
                                               0.5 * (next_pt[cno] - in_pt[cno]))

        el_no = cur_voxel_coords[0] * x_slice_size + \
                cur_voxel_coords[1] * vd[2] + cur_voxel_coords[2]

        # Use + 1 since the first track would be ignored
        if touched_tags_v[el_no] != track_idx + 1:
            touched_tags_v[el_no] = track_idx + 1
            traversal_tags_v[el_no] += 1

    np.seterr(**flags)
    return traversal_tags.reshape(vol_dims)
