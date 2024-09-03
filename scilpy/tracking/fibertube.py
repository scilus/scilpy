import numpy as np

from math import sqrt
from numba import njit
from scilpy.io.utils import v_enumerate


from scilpy.io.utils import add_overwrite_arg


def add_mandatory_tracking_options(p):
    p.add_argument('in_centerlines',
                   help='Path to the tractogram file containing the \n'
                        'fibertube centerlines (must be .trk or .tck). \n'
                        'This tractogram must be void of any intersection \n'
                        'given the diameter supplied for each streamline \n'
                        '(see scil_filter_intersections.py).')

    p.add_argument('in_diameters',
                   help='Path to a text file containing a list of the \n'
                        'diameters of each fibertube in mm (.txt). Each \n'
                        'line corresponds to the identically numbered \n'
                        'streamline.')

    p.add_argument('in_mask',
                   help='Tracking mask (.nii.gz).\n'
                        'Tracking will stop outside this mask. The last point '
                        'of each \nstreamline (triggering the stopping '
                        'criteria) IS added to the streamline.')

    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')

    p.add_argument('step_size', type=float,
                   help='Step size of the tracking algorithm, in mm.')

    p.add_argument('sampling_radius', type=float,
                   help='Radius of the circular region from which the \n'
                        'algorithm will determine the next direction.')


def add_tracking_options(p):
    track_g = p.add_argument_group('Tracking options')
    track_g.add_argument(
        '--min_length', type=float, default=10.,
        metavar='m',
        help='Minimum length of a streamline in mm. '
        '[%(default)s]')
    track_g.add_argument(
        '--max_length', type=float, default=300.,
        metavar='M',
        help='Maximum length of a streamline in mm. '
        '[%(default)s]')
    track_g.add_argument(
        '--theta', type=float, default=60.,
        help='Maximum angle between 2 steps. If the angle is '
             'too big, streamline is \nstopped and the '
             'following point is NOT included.\n'
             '[%(default)s]')
    track_g.add_argument(
        '--rk_order', metavar="K", type=int, default=1,
        choices=[1, 2, 4],
        help="The order of the Runge-Kutta integration used \n"
             'for the step function. \n'
             'For more information, refer to the note in the \n'
             'script description. [%(default)s]')
    track_g.add_argument(
        '--max_invalid_nb_points', metavar='MAX', type=int,
        default=0,
        help='Maximum number of steps without valid \n'
             'direction, \nex: if threshold on ODF or max \n'
             'angles are reached. \n'
             'Default: 0, i.e. do not add points following '
             'an invalid direction.')
    track_g.add_argument(
        '--forward_only', action='store_true',
        help='If set, tracks in one direction only (forward) \n'
             'given the \ninitial seed.')
    track_g.add_argument(
        '--keep_last_out_point', action='store_true',
        help='If set, keep the last point (once out of the \n'
             'tracking mask) of the streamline. Default: discard \n'
             'them. This is the default in Dipy too. \n'
             'Note that points obtained after an invalid direction \n'
             '(based on the propagator\'s definition of invalid) \n'
             'are never added.')


def add_seeding_options(p):
    seed_group = p.add_argument_group(
        'Seeding options',
        'When no option is provided, uses --nb_seeds_per_fiber 5.')
    seed_group.add_argument(
        '--nb_seeds_per_fiber', type=int, default=5,
        help='The number of seeds planted in the first segment \n'
             'of each fiber. The total amount of streamlines will \n'
             'be [nb_seeds_per_fiber] * [nb_fibers]. [%(default)s]')
    seed_group.add_argument(
        '--nb_fibers', type=int,
        help='If set, the script will only track a specified \n'
             'amount of fibers. Otherwise, the entire tractogram \n'
             'will be tracked. The total amount of streamlines \n'
             'will be [nb_seeds_per_fiber] * [nb_fibers].')


def add_out_options(p):
    out_g = p.add_argument_group('Output options')
    out_g.add_argument(
        '--do_not_compress', action='store_true',
        help='If set, streamlines will not be compressed as \n'
             'they are saved. Compression is activated by default because \n'
             'of the excessive coordinate density of the output \n'
             'tractogram. Only deactivate for benchmarking or if \n'
             'you will compress later.')
    out_g.add_argument(
        '--save_seeds', action='store_true',
        help='If set, the seeds used for tracking will be saved \n'
             'in an additional .txt file. Its file name is derived \n'
             'from the out_tractogram parameter with "_seeds" \n'
             'appended.')
    out_g.add_argument(
        '--save_config', action='store_true',
        help='If set, some parameters used for tracking will be saved \n'
             'in an additional .txt file. Its file name is derived \n'
             'from the out_tractogram parameter with "_config" \n'
             'appended.')

    add_overwrite_arg(out_g)


def segment_tractogram(streamlines, verbose=False):
    """
    Separates all streamlines of a tractogram into segments that connect
    each position. Then, flattens the resulting 2D array and returns it

    Parameters
    ----------
    streamlines : list
        Streamlines to segment. This function is compatible with streamlines
        as a fixed array, as long as the padding value is a number. Padding
        will also be present in the result value.

    Returns
    -------
    A tuple containing the following values:
        centers : ndarray[float]
            A flattened array of all the tractogram's segment centers
        indices : ndarray[Tuple[int, int]]
            A flattened array of all the tractogram's segment indices
        max_length: float
            Length of the longest segment of the whole tractogram
    """
    centers = []
    indices = []
    max_length = 0.
    for si, s in v_enumerate(streamlines, verbose):
        centers.append((s[1:] + s[:-1]) / 2)
        indices.append([(si, pi) for pi in range(len(s)-1)])

        max_length_candidate = np.amax(np.linalg.norm(s[1:] - s[:-1], axis=-1))

        if max_length_candidate > max_length:
            max_length = float(max_length_candidate)

    centers = np.vstack(centers)
    indices = np.vstack(indices)

    return (centers, indices, max_length)


@njit
def rotation_between_vectors_matrix(vec1, vec2):
    """
    Rotation matrix that aligns vec1 to vec2. Numba compatible.

    https://math.stackexchange.com/questions/180418/calculate-
    rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    Parameters
    ----------
    origin : any
    destination : any
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.

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
    amount: int
        Amount of samples to be produced.

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
def sample_cylinder(center, axis, radius: float, length: float,
                    sample_count: int, random_generator: np.random.Generator):
    """
    Samples a cylinder uniformly given its dimensions and the amount of
    samples.

    Parameters
    ----------
    center: ndarray
        Center coordinates of the cylinder
    axis: ndarray
        Center axis of the cylinder, in the form of a vector. Does not have to
        be normalized.
    radius: float
    length: float
    sample_count: int
        Amount of samples to be produced.

    Returns
    -------
    samples: list
        Array containing the coordinates of each sample.
    """
    samples = []
    while (len(samples) < sample_count):
        reference = np.array([0., 0., 1.], dtype=axis.dtype)
        axis /= np.linalg.norm(axis)
        half_length = length / 2

        # Generation
        x = random_generator.uniform(-radius, radius)
        y = random_generator.uniform(-radius, radius)
        z = random_generator.uniform(-half_length, half_length)
        sample = np.array([x, y, z], dtype=np.float32)

        # Rotation
        rotation_matrix = np.eye(4, dtype=np.float32)
        rotation_matrix[:3, :3] = rotation_between_vectors_matrix(
            reference,
            axis).astype(np.float32)
        sample = np.dot(rotation_matrix, np.append(sample, np.float32(1.)))[:3]

        # Translation
        sample += center

        bottom_center = center - axis * half_length
        top_center = center + axis * half_length
        if (point_in_cylinder(bottom_center, top_center, radius, sample)):
            samples.append(sample)
    return samples


@njit
def point_in_cylinder(pt1, pt2, r, q):
    vec = np.subtract(pt2, pt1)
    cond_1 = np.dot(q - pt1, vec) >= 0
    cond_2 = np.dot(q - pt2, vec) <= 0
    cond_3 = (np.linalg.norm(np.cross(q - pt1, vec)) /
              np.linalg.norm(vec)) <= r
    return cond_1 and cond_2 and cond_3


@njit
def sphere_cylinder_intersection(sph_p, sph_r: float, cyl_p1, cyl_p2,
                                 cyl_r: float, sample_count: int,
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

    Returns
    -------
    inter_volume: float
        Approximate volume of the sphere-cylinder intersection.
    is_estimated: boolean
        Indicates whether or not the resulting volume has been estimated.
        If true, inter_volume has a probability of error due to sample count.
    """
    cyl_axis = cyl_p2 - cyl_p1
    cyl_length = np.linalg.norm(cyl_axis)
    cyl_center = (cyl_p1 + cyl_p2) / 2

    # If cylinder is completely inside the sphere.
    if (np.linalg.norm(sph_p - cyl_p1) + cyl_r <= sph_r and
            np.linalg.norm(sph_p - cyl_p2) + cyl_r <= sph_r):
        cyl_volume = np.pi * (cyl_r ** 2) * cyl_length
        return cyl_volume, False

    # If cylinder is completely outside the sphere.
    _, vector, _, _ = dist_point_segment(cyl_p1, cyl_p2, sph_p)
    if np.linalg.norm(vector) >= sph_r + cyl_r:
        return 0, False

    # High probability of intersection.
    samples = sample_cylinder(cyl_center, cyl_axis, cyl_r, cyl_length,
                              sample_count, random_generator)

    inter_samples = 0
    for sample in samples:
        if np.linalg.norm(sph_p - sample) < sph_r:
            inter_samples += 1

    # Proportion of cylinder samples common to both shapes * cylinder volume.
    cyl_volume = np.pi * (cyl_r ** 2) * cyl_length
    inter_volume = (inter_samples / sample_count) * cyl_volume

    return inter_volume, True


@njit
def create_perpendicular(v):
    vp = np.array([0., 0., 0.])
    for m in range(3):
        if v[m] == 0.:
            continue
        n = (m + 1) % 3
        vp[n] = -v[m]
        vp[m] = -v[n]

    return vp / np.linalg.norm(vp)


@njit
def dist_point_segment(P0, P1, Q):
    return dist_segment_segment(P0, P1, Q, Q)


@njit
def dist_segment_segment(P0, P1, Q0, Q1):
    """
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
    diff = Ps - Qt
    sqr_distance = np.dot(diff, diff)
    distance = sqrt(sqr_distance)
    return (distance, diff, Ps, Qt)
