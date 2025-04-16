import numpy as np

from math import sqrt
from numba import njit
from typing import Literal
from scilpy.tracking.utils import tqdm_if_verbose


def streamlines_to_segments(streamlines, streamlines_length=None, verbose=False):
    """
    Separates all streamlines of a tractogram into segments that connect
    each position. Then, flattens the resulting 2D array and returns it

    Parameters
    ----------
    streamlines : list
        Streamlines to segment. This function is compatible with streamlines
        as a fixed array with a padding value, as long as the
        [streamlines_length] parameter is given. Padding will be kept in the
        result value.
    streamlines_length: list
        Length of each streamline. Only necessary if streamlines are given
        as a fixed array.
    verbose: bool
        Wether the function should be verbose.

    Returns
    -------
    centers : ndarray[float]
        A flattened array of all the tractogram's segment centers
    indices : ndarray[Tuple[int, int]]
        A flattened array of all the tractogram's segment indices
    max_length: float
        Length of the longest segment of the whole tractogram
    """
    centers = []
    indices = []
    max_seg_length = 0.

    for si, s in tqdm_if_verbose(enumerate(streamlines), verbose,
                                 total=len(streamlines)):
        centers.append((s[1:] + s[:-1]) / 2)
        indices.append([(si, pi) for pi in range(len(s)-1)])

        if streamlines_length is None:
            max_seg_length_candidate = np.amax(
                np.linalg.norm(s[1:] - s[:-1], axis=-1))
        else:
            length = streamlines_length[si]
            max_seg_length_candidate = np.amax(
                np.linalg.norm(s[1:length] - s[:length-1], axis=-1))

        if max_seg_length_candidate > max_seg_length:
            max_seg_length = float(max_seg_length_candidate)

    centers = np.vstack(centers)
    indices = np.vstack(indices)

    return (centers, indices, max_seg_length)


@njit
def rotation_between_vectors_matrix(vec1, vec2):
    """
    Produces a rotation matrix that aligns a 3D vector 'vec1' with another 3D
    vector 'vec2'. Numba compatible.

    https://math.stackexchange.com/questions/180418/calculate-
    rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    Parameters
    ----------
    vec1: ndarray
        Vector to be rotated
    vec2: ndarray
        Targeted orientation

    Returns
    -------
    rotation_matrix: ndarray
        A transform matrix (3x3) which when applied to vec1, aligns it with
        vec2.
    """
    a, b = ((vec1 / np.linalg.norm(vec1)).reshape(3),
            (vec2 / np.linalg.norm(vec2)).reshape(3))
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s != 0:
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + (kmat +
                                       kmat.dot(kmat) * ((1 - c) / (s ** 2)))
    else:
        rotation_matrix = np.eye(3)
    return rotation_matrix


@njit
def sample_sphere(center, radius: float, amount: int,
                  rand_gen: np.random.Generator):
    """
    Samples a sphere uniformly given its dimensions and the amount of samples.

    Parameters
    ----------
    center: ndarray
        Center coordinates of the sphere. Can be [0, 0, 0] if only the
        relative displacement interests you.
    radius: float
        Radius of the sphere.
    amount: int
        Amount of samples to be produced.
    rand_gen: numpy random generator
        Numpy random generator used for producing samples within the sphere.

    Returns
    -------
    samples: list
        Array containing the coordinates of each sample.
    """
    samples = []
    while (len(samples) < amount):
        sample = np.array([rand_gen.uniform(-radius, radius),
                           rand_gen.uniform(-radius, radius),
                           rand_gen.uniform(-radius, radius)])
        if np.linalg.norm(sample) <= radius:
            samples.append(sample + center)
    return samples


@njit
def sample_cylinder(pt1, pt2, radius: float, sample_count: int,
                    random_generator: np.random.Generator):
    """
    Samples a cylinder uniformly given its dimensions and the amount of
    samples.

    Parameters
    ----------
    pt1: ndarray
        First extremity of the cylinder axis
    pt2: ndarray
        Second extremity of the cylinder axis
    radius: float
        Radius of the cylinder.
    sample_count: int
        Amount of samples to be produced.
    rand_gen: numpy random generator
        Numpy random generator used for producing samples within the sphere.

    Returns
    -------
    samples: list
        Array containing the coordinates of each sample.
    """
    samples = []
    while (len(samples) < sample_count):
        axis = pt2 - pt1
        center = (pt1 + pt2) / 2
        half_length = np.linalg.norm(axis) / 2
        axis /= np.linalg.norm(axis)
        reference = np.array([0., 0., 1.], dtype=axis.dtype)

        # Generation
        x = random_generator.uniform(-radius, radius)
        y = random_generator.uniform(-radius, radius)
        z = random_generator.uniform(-half_length, half_length)
        sample = np.array([x, y, z], dtype=np.float64)

        # Rotation
        rotation_matrix = np.eye(4, dtype=np.float64)
        rotation_matrix[:3, :3] = rotation_between_vectors_matrix(
            reference,
            axis).astype(np.float32)
        sample = np.dot(rotation_matrix, np.append(sample, 1.))[:3]

        # Translation
        sample += center
        sample = sample.astype(np.float32)

        if (point_in_cylinder(pt1, pt2, radius, sample)):
            samples.append(sample)
    return samples


@njit
def point_in_cylinder(pt1, pt2, r, q):
    vec = pt2 - pt1
    cond_1 = np.dot(q - pt1, vec) >= 0
    cond_2 = np.dot(q - pt2, vec) <= 0
    cond_3 = (np.linalg.norm(np.cross(q - pt1, vec)) /
              np.linalg.norm(vec)) <= r
    return cond_1 and cond_2 and cond_3


@njit
def sphere_cylinder_intersection(sph_p, sph_r: float, cyl_p1, cyl_p2,
                                 cyl_r: float, sample_count: int,
                                 shape_to_sample: Literal["sphere",
                                                          "cylinder"],
                                 random_generator: np.random.Generator):
    """
    Estimates the volume of intersection between a cylinder and a sphere by
    sampling the cylinder. Most efficient when the cylinder is smaller than
    the sphere.

    Parameters
    ----------
    sph_p: ndarray
        Center coordinate of the sphere.
    sph_r: float
        Radius of the sphere.
    cyl_p1: ndarray
        First point of the cylinder's center segment.
    cyl_p2: ndarray
        Second point of the cylinder's center segment.
    cyl_r: float
        Radius of the cylinder.
    sample_count: int
        Amount of samples to use for the estimation.
    shape_to_sample: 'sphere' | 'cylinder'
        Shape to sample. For best accuracy, this should be the smallest shape.
    random_generator: numpy random generator
        Numpy random generator used for producing sample coordinates.

    Returns
    -------
    inter_volume: float
        Approximate volume of the sphere-cylinder intersection.
    is_estimated: boolean
        Indicates whether or not the resulting volume has been estimated using
        samples instead of an analytical solution.
    """
    cyl_axis = cyl_p2 - cyl_p1
    cyl_length = np.linalg.norm(cyl_axis)

    # If cylinder is completely inside the sphere.
    if (np.linalg.norm(sph_p - cyl_p1) + cyl_r <= sph_r and
            np.linalg.norm(sph_p - cyl_p2) + cyl_r <= sph_r):
        cyl_volume = np.pi * (cyl_r ** 2) * cyl_length
        return cyl_volume, False

    # If cylinder is completely outside the sphere.
    _, vector, _ = dist_point_segment(cyl_p1, cyl_p2, sph_p)
    if np.linalg.norm(vector) >= sph_r + cyl_r:
        return 0, False

    if shape_to_sample == "cylinder":
        samples = sample_cylinder(cyl_p1, cyl_p2, cyl_r, sample_count,
                                  random_generator)

        inter_samples = 0
        # Count the cylinder samples that also land in the sphere.
        for sample in samples:
            if np.linalg.norm(sph_p - sample) < sph_r:
                inter_samples += 1

        cyl_volume = np.pi * (cyl_r ** 2) * cyl_length
        inter_volume = (inter_samples / sample_count) * cyl_volume
    elif shape_to_sample == "sphere":
        samples = sample_sphere(sph_p, sph_r, sample_count,
                                random_generator)

        inter_samples = 0
        # Count the sphere samples that also land in the cylinder.
        for sample in samples:
            if dist_point_segment(cyl_p1, cyl_p2, sample)[0] < cyl_r:
                inter_samples += 1

        sph_volume = 4 * np.pi * (sph_r ** 3) / 3
        inter_volume = (inter_samples / sample_count) * sph_volume
    else:
        raise ValueError("The provided shape_to_sample parameter is not "
                         "one of the expected choices.")

    return inter_volume, True


@njit
def create_perpendicular(v: np.ndarray):
    """
    Generates a vector perpendicular to v.

    Source: https://math.stackexchange.com/questions/133177/finding-a-unit-
            vector-perpendicular-to-another-vector
    Answer by Ahmed Fasih

    Parameters
    ----------
    v: ndarray
        Vector from which a perpendicular vector will be generated.

    Returns
    -------
    vp: ndarray
        Vector perpendicular to v.
    """
    vp = np.array([0., 0., 0.])
    if v.all() == vp.all():
        return vp
    for m in range(3):
        if v[m] == 0.:
            continue
        n = (m + 1) % 3
        vp[n] = -v[m]
        vp[m] = -v[n]

    return vp / np.linalg.norm(vp)


@njit
def dist_point_segment(p0, p1, q):
    """
    Calculates the shortest distance between a 3D point q and a segment p0-p1.

    Parameters
    ----------
    p0: ndarray
        Point forming the first end of the segment.
    p1: ndarray
        Point forming the second end of the segment.
    q: ndarray
        Point coordinates.

    Returns
    -------
    distance: float
        Shortest distance between the two segments
    v: ndarray
        Vector representing the distance between the two segments.
        v = Ps - q and |v| = distance
    Ps: ndarray
        Point coordinates on segment P that is closest to point q
    """
    return dist_segment_segment(p0, p1, q, q)[:3]


@njit
def dist_segment_segment(P0, P1, Q0, Q1):
    """
    Calculates the shortest distance between two 3D segments P0-P1 and Q0-Q1.

    Parameters
    ----------
    P0: ndarray
        Point forming the first end of the P segment.
    P1: ndarray
        Point forming the second end of the P segment.
    Q0: ndarray
        Point forming the first end of the Q segment.
    Q1: ndarray
        Point forming the second end of the Q segment.

    Returns
    -------
    distance: float
        Shortest distance between the two segments
    v: ndarray
        Vector representing the distance between the two segments.
        v = Ps - Qt and |v| = distance
    Ps: ndarray
        Point coordinates on segment P that is closest to segment Q
    Qt: ndarray
        Point coordinates on segment Q that is closest to segment P

    This function is a python version of the following code:
    https://www.geometrictools.com/GTE/Mathematics/DistSegmentSegment.h

    Scientific source:
    https://www.geometrictools.com/Documentation/DistanceLine3Line3.pdf

    """
    P1mP0 = np.subtract(P1, P0)
    Q1mQ0 = np.subtract(Q1, Q0)
    P0mQ0 = np.subtract(P0, Q0)

    a = np.dot(P1mP0, P1mP0)
    b = np.dot(P1mP0, Q1mQ0)
    c = np.dot(Q1mQ0, Q1mQ0)
    d = np.dot(P1mP0, P0mQ0)
    e = np.dot(Q1mQ0, P0mQ0)
    det = a * c - b * b
    s = t = nd = bmd = bte = ctd = bpe = ate = btd = None

    if det > 0:
        bte = b * e
        ctd = c * d
        if bte <= ctd:  # s <= 0
            s = 0
            if e <= 0:  # t <= 0
                # section 6
                t = 0
                nd = -d
                if nd >= a:
                    s = 1
                elif nd > 0:
                    s = nd / a
                # else: s is already 0
            elif e < c:  # 0 < t < 1
                # section 5
                t = e / c
            else:  # t >= 1
                # section 4
                t = 1
                bmd = b - d
                if bmd >= a:
                    s = 0
                elif bmd > 0:
                    s = bmd / a
                # else:  s is already 0
        else:  # s > 0
            s = bte - ctd
            if s >= det:  # s >= 1
                # s = 1
                s = 1
                bpe = b + e
                if bpe <= 0:  # t <= 0
                    # section 8
                    t = 0
                    nd = -d
                    if nd <= 0:
                        s = 0
                    elif nd < a:
                        s = nd / a
                    # else: s is already 1
                elif bpe < c:  # 0 < t < 1
                    # section 1
                    t = bpe / c
                else:  # t >= 1
                    # section 2
                    t = 1
                    bmd = b - d
                    if bmd <= 0:
                        s = 0
                    elif bmd < a:
                        s = bmd / a
                    # else:  s is already 1
            else:  # 0 < s < 1
                ate = a * e
                btd = b * d
                if ate <= btd:  # t <= 0
                    # section 7
                    t = 0
                    nd = -d
                    if nd <= 0:
                        s = 0
                    elif nd >= a:
                        s = 1
                    else:
                        s = nd / a
                else:  # t > 0
                    t = ate - btd
                    if t >= det:  # t >= 1
                        # section 3
                        t = 1
                        bmd = b - d
                        if bmd <= 0:
                            s = 0
                        elif (bmd >= a):
                            s = 1
                        else:
                            s = bmd / a
                    else:  # 0 < t < 1
                        # section 0
                        s /= det
                        t /= det

    else:
        # The segments are parallel. The quadratic factors to
        #   R(s,t) = a*(s-(b/a)*t)^2 + 2*d*(s - (b/a)*t) + f
        # where a*c = b^2, e = b*d/a, f = |P0-Q0|^2, and b is not
        # 0. R is constant along lines of the form s-(b/a)*t = k
        # and its occurs on the line a*s - b*t + d = 0. This line
        # must intersect both the s-axis and the t-axis because 'a'
        # and 'b' are not 0. Because of parallelism, the line is
        # also represented by -b*s + c*t - e = 0.
        #
        # The code determines an edge of the domain [0,1]^2 that
        # intersects the minimum line, or if n1 of the edges
        # intersect, it determines the closest corner to the minimum
        # line. The conditionals are designed to test first for
        # intersection with the t-axis (s = 0) using
        # -b*s + c*t - e = 0 and then with the s-axis (t = 0) using
        # a*s - b*t + d = 0.

        # When s = 0, solve c*t - e = 0 (t = e/c).
        if e <= 0:  # t <= 0
            # Now solve a*s - b*t + d = 0 for t = 0 (s = -d/a).
            t = 0
            nd = -d
            if nd <= 0:  # s <= 0
                # section 6
                s = 0
            elif nd >= a:  # s >= 1
                # section 8
                s = 1
            else:  # 0 < s < 1
                # section 7
                s = nd / a
        elif e >= c:  # t >= 1
            # Now solve a*s - b*t + d = 0 for t = 1 (s = (b-d)/a).
            t = 1
            bmd = b - d
            if bmd <= 0:  # s <= 0
                # section 4
                s = 0
            elif bmd >= a:  # s >= 1
                # section 2
                s = 1
            else:  # 0 < s < 1
                # section 3
                s = bmd / a
        else:  # 0 < t < 1
            # The point (0,e/c) is on the line and domain, so we have
            # 1 point at which R is a minimum.
            s = 0
            t = e / c

    Ps = P0 + s * P1mP0
    Qt = Q0 + t * Q1mQ0
    v = Ps - Qt
    sqr_distance = np.dot(v, v)
    distance = sqrt(sqr_distance)
    return (distance, v, Ps, Qt)
