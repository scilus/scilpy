# -*- coding: utf-8 -*-
import logging
from typing import Iterable

import nibabel as nib
import numpy as np
from nibabel.streamlines import TrkFile
from nibabel.streamlines.tractogram import LazyTractogram, TractogramItem
from tqdm import tqdm

from dipy.core.sphere import HemiSphere
from dipy.data import get_sphere
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter, PTTDirectionGetter)
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.utils import create_tractogram_header, get_reference_info
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.tracking.streamlinespeed import compress_streamlines, length
from scilpy.io.utils import (add_compression_arg, add_overwrite_arg,
                             add_sh_basis_args)
from scilpy.reconst.utils import find_order_from_nb_coeff, get_maximas


class TrackingDirection(list):
    """
    Tracking direction use as 3D cartesian direction (list(x,y,z))
    and has an index to work with discrete sphere.
    """

    def __init__(self, cartesian, index=None):
        super(TrackingDirection, self).__init__(cartesian)
        self.index = index


def add_mandatory_options_tracking(p):
    """
    Args that are required in both scil_tracking_local and
    scil_tracking_local_dev scripts.
    """
    p.add_argument('in_odf',
                   help='File containing the orientation diffusion function \n'
                        'as spherical harmonics file (.nii.gz). Ex: ODF or '
                        'fODF.')
    p.add_argument('in_seed',
                   help='Seeding mask (.nii.gz).')
    p.add_argument('in_mask',
                   help='Tracking mask (.nii.gz).\n'
                        'Tracking will stop outside this mask. The last point '
                        'of each \nstreamline (triggering the stopping '
                        'criteria) IS added to the streamline.')
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')


def add_tracking_options(p):
    """
    Options that are available in both scil_tracking_local and
    scil_tracking_local_dev scripts.
    """
    track_g = p.add_argument_group('Tracking options')
    track_g.add_argument('--step', dest='step_size', type=float, default=0.5,
                         help='Step size in mm. [%(default)s]')
    track_g.add_argument('--min_length', type=float, default=10.,
                         metavar='m',
                         help='Minimum length of a streamline in mm. '
                              '[%(default)s]')
    track_g.add_argument('--max_length', type=float, default=300.,
                         metavar='M',
                         help='Maximum length of a streamline in mm. '
                              '[%(default)s]')
    track_g.add_argument('--theta', type=float,
                         help='Maximum angle between 2 steps. If the angle is '
                              'too big, streamline is \nstopped and the '
                              'following point is NOT included.\n'
                              '["eudx"=60, "det"=45, "prob"=20, "ptt"=20]')
    track_g.add_argument('--sfthres', dest='sf_threshold', metavar='sf_th',
                         type=float, default=0.1,
                         help='Spherical function relative threshold. '
                              '[%(default)s]')
    add_sh_basis_args(track_g)

    return track_g


def add_tracking_ptt_options(p):
    track_g = p.add_argument_group('PTT options')
    track_g.add_argument('--probe_length', dest='probe_length',
                         type=float, default=1.0,
                         help='The length of the probes. Smaller value\n'
                              'yields more dispersed fibers. [%(default)s]')
    track_g.add_argument('--probe_radius', dest='probe_radius',
                         type=float, default=0,
                         help='The radius of the probe. A large probe_radius\n'
                              'helps mitigate noise in the pmf but it might\n'
                              'make it harder to sample thin and intricate\n'
                              'connections, also the boundary of fiber\n'
                              'bundles might be eroded. [%(default)s]')
    track_g.add_argument('--probe_quality', dest='probe_quality',
                         type=int, default=3,
                         help='The quality of the probe. This parameter sets\n'
                              'the number of segments to split the cylinder\n'
                              'along the length of the probe (minimum=2) '
                              '[%(default)s]')
    track_g.add_argument('--probe_count', dest='probe_count',
                         type=int, default=1,
                         help='The number of probes. This parameter sets the\n'
                              'number of parallel lines used to model the\n'
                              'cylinder (minimum=1). [%(default)s]')
    track_g.add_argument('--support_exponent',
                         type=float, default=3,
                         help='Data support exponent, used for rejection\n'
                              'sampling. [%(default)s]')

    return track_g


def add_seeding_options(p):
    """
    Options that are available in both scil_tracking_local and
    scil_tracking_local_dev scripts.
    """
    seed_group = p.add_argument_group(
        'Seeding options', 'When no option is provided, uses --npv 1.')
    seed_sub_exclusive = seed_group.add_mutually_exclusive_group()
    seed_sub_exclusive.add_argument('--npv', type=int,
                                    help='Number of seeds per voxel.')
    seed_sub_exclusive.add_argument('--nt', type=int,
                                    help='Total number of seeds to use.')


def add_out_options(p):
    """
    Options that are available in both scil_tracking_local and
    scil_tracking_local_dev scripts.
    """
    out_g = p.add_argument_group('Output options')
    msg = ("\nA rule of thumb is to set it to 0.1mm for deterministic \n"
           "streamlines and to 0.2mm for probabilitic streamlines.")
    add_compression_arg(out_g, additional_msg=msg)

    add_overwrite_arg(out_g)
    out_g.add_argument('--save_seeds', action='store_true',
                       help='If set, save the seeds used for the tracking \n '
                            'in the data_per_streamline property.\n'
                            'Hint: you can then use '
                            'scilpy_compute_seed_density_map.')
    return out_g


def verify_streamline_length_options(parser, args):
    if not args.min_length >= 0:
        parser.error('min_length must be >= 0, but {}mm was provided.'
                     .format(args.min_length))
    if args.max_length < args.min_length:
        parser.error('max_length must be > than min_length, but '
                     'min_length={}mm and max_length={}mm.'
                     .format(args.min_length, args.max_length))


def verify_seed_options(parser, args):
    if args.npv and args.npv <= 0:
        parser.error('Number of seeds per voxel must be > 0.')

    if args.nt and args.nt <= 0:
        parser.error('Total number of seeds must be > 0.')


def tqdm_if_verbose(generator: Iterable, verbose: bool, *args, **kwargs):
    if verbose:
        return tqdm(generator, *args, **kwargs)
    return generator


def save_tractogram(
        streamlines_generator, tracts_format, ref_img, total_nb_seeds,
        out_tractogram, min_length, max_length, compress, save_seeds, verbose
):
    """ Save the streamlines on-the-fly using a generator. Tracts are
    filtered according to their length and compressed if requested. Seeds
    are saved if requested. The tractogram is shifted and scaled according
    to the file format.

    Parameters
    ----------
    streamlines_generator : generator
        Streamlines generator.
    tracts_format : TrkFile or TckFile
        Tractogram format.
    ref_img : nibabel.Nifti1Image
        Image used as reference.
    total_nb_seeds : int
        Total number of seeds.
    out_tractogram : str
        Output tractogram filename.
    min_length : float
        Minimum length of a streamline in mm.
    max_length : float
        Maximum length of a streamline in mm.
    compress : float
        Distance threshold for compressing streamlines in mm.
    save_seeds : bool
        If True, save the seeds used for the tracking in the
        data_per_streamline property.
    verbose : bool
        If True, display progression bar.

    """

    voxel_size = ref_img.header.get_zooms()[0]

    scaled_min_length = min_length / voxel_size
    scaled_max_length = max_length / voxel_size

    # Tracking is expected to be returned in voxel space, origin `center`.
    def tracks_generator_wrapper():
        for strl, seed in tqdm_if_verbose(streamlines_generator,
                                          verbose=verbose,
                                          total=total_nb_seeds,
                                          miniters=int(total_nb_seeds / 100),
                                          leave=False):
            if (scaled_min_length <= length(strl) <= scaled_max_length):
                # Seeds are saved with origin `center` by our own convention.
                # Other scripts (e.g. scil_compute_seed_density_map) expect so.
                dps = {}
                if save_seeds:
                    dps['seeds'] = seed

                if compress:
                    # compression threshold is given in mm, but we
                    # are in voxel space
                    strl = compress_streamlines(
                        strl, compress / voxel_size)

                # TODO: Use nibabel utilities for dealing with spaces
                if tracts_format is TrkFile:
                    # Streamlines are dumped in mm space with
                    # origin `corner`. This is what is expected by
                    # LazyTractogram for .trk files (although this is not
                    # specified anywhere in the doc)
                    strl += 0.5
                    strl *= voxel_size  # in mm.
                else:
                    # Streamlines are dumped in true world space with
                    # origin center as expected by .tck files.
                    strl = np.dot(strl, ref_img.affine[:3, :3]) + \
                        ref_img.affine[:3, 3]

                yield TractogramItem(strl, dps, {})

    tractogram = LazyTractogram.from_data_func(tracks_generator_wrapper)
    tractogram.affine_to_rasmm = ref_img.affine

    filetype = nib.streamlines.detect_format(out_tractogram)
    reference = get_reference_info(ref_img)
    header = create_tractogram_header(filetype, *reference)

    # Use generator to save the streamlines on-the-fly
    nib.streamlines.save(tractogram, out_tractogram, header=header)


def get_direction_getter(in_img, algo, sphere, sub_sphere, theta, sh_basis,
                         voxel_size, sf_threshold, sh_to_pmf,
                         probe_length, probe_radius, probe_quality,
                         probe_count, support_exponent, is_legacy=True):
    """ Return the direction getter object.

    Parameters
    ----------
    in_img: str
        Path to the input odf file.
    algo: str
        Algorithm to use for tracking. Can be 'det', 'prob', 'ptt' or 'eudx'.
    sphere: str
        Name of the sphere to use for tracking.
    sub_sphere: int
        Number of subdivisions to use for the sphere.
    theta: float
        Angle threshold for tracking.
    sh_basis: str
        Name of the sh basis to use for tracking.
    voxel_size: float
        Voxel size of the input data.
    sf_threshold: float
        Spherical function-amplitude threshold for tracking.
    sh_to_pmf: bool
        Map sherical harmonics to spherical function (pmf) before tracking
        (faster, requires more memory).
    probe_length : float
        The length of the probes. Shorter probe_length
        yields more dispersed fibers.
    probe_radius : float
        The radius of the probe. A large probe_radius
        helps mitigate noise in the pmf but it might
        make it harder to sample thin and intricate
        connections, also the boundary of fiber
        bundles might be eroded.
    probe_quality : int
        The quality of the probe. This parameter sets
        the number of segments to split the cylinder
        along the length of the probe (minimum=2).
    probe_count : int
        The number of probes. This parameter sets the
        number of parallel lines used to model the
        cylinder (minimum=1).
    support_exponent : float
        Data support exponent, used for rejection sampling.
    is_legacy : bool, optional
        Whether or not the SH basis is in its legacy form.

    Return
    ------
    dg: dipy.direction.DirectionGetter
        The direction getter object.
    """
    img_data = nib.load(in_img).get_fdata(dtype=np.float32)

    sphere = HemiSphere.from_sphere(
        get_sphere(sphere)).subdivide(sub_sphere)

    # Theta depends on user choice and algorithm
    theta = get_theta(theta, algo)

    # Heuristic to find out if the input are peaks or fodf
    # fodf are always around 0.15 and peaks around 0.75
    # Peaks have more zero values than fodf. The first value of fodf is
    # usually the highest.
    non_zeros_count = np.count_nonzero(np.sum(img_data, axis=-1))
    non_first_val_count = np.count_nonzero(np.argmax(img_data, axis=-1))
    is_peaks = non_first_val_count / non_zeros_count > 0.5

    if algo in ['det', 'prob', 'ptt']:
        if is_peaks:
            logging.warning(
                'Input detected as peaks. Input should be fodf for '
                'det/prob/ptt, verify input just in case.')
        kwargs = {}
        if algo == 'ptt':
            dg_class = PTTDirectionGetter
            # Considering the step size usually used, the probe length
            # can be set as the voxel size.
            kwargs = {'probe_length': probe_length,
                      'probe_radius': probe_radius,
                      'probe_quality': probe_quality,
                      'probe_count': probe_count,
                      'data_support_exponent': support_exponent}
        elif algo == 'det':
            dg_class = DeterministicMaximumDirectionGetter
        else:
            dg_class = ProbabilisticDirectionGetter
        return dg_class.from_shcoeff(
            shcoeff=img_data, max_angle=theta, sphere=sphere,
            basis_type=sh_basis, legacy=is_legacy, sh_to_pmf=sh_to_pmf,
            relative_peak_threshold=sf_threshold, **kwargs)
    elif algo == 'eudx':
        # Code for algo EUDX. We don't use peaks_from_model
        # because we want the peaks from the provided sh.
        img_shape_3d = img_data.shape[:-1]
        dg = PeaksAndMetrics()
        dg.sphere = sphere
        dg.ang_thr = theta
        dg.qa_thr = sf_threshold

        if is_peaks:
            # If the input is peaks, we compute their amplitude and
            # find the closest direction on the sphere.
            logging.info('Input detected as peaks.')
            nb_peaks = img_data.shape[-1] // 3
            slices = np.arange(0, 15 + 1, 3)
            peak_values = np.zeros(img_shape_3d + (nb_peaks,))
            peak_indices = np.zeros(img_shape_3d + (nb_peaks,))

            for idx in np.argwhere(np.sum(img_data, axis=-1)):
                idx = tuple(idx)
                for i in range(nb_peaks):
                    peak_values[idx][i] = np.linalg.norm(
                        img_data[idx][slices[i]:slices[i + 1]], axis=-1)
                    peak_indices[idx][i] = sphere.find_closest(
                        img_data[idx][slices[i]:slices[i + 1]])

            dg.peak_dirs = img_data
        else:
            # If the input is not peaks, we assume it is fodf
            # and we compute the peaks from the fodf.
            logging.info('Input detected as fodf.')
            npeaks = 5
            peak_dirs = np.zeros((img_shape_3d + (npeaks, 3)))
            peak_values = np.zeros((img_shape_3d + (npeaks,)))
            peak_indices = np.full((img_shape_3d + (npeaks,)), -1,
                                   dtype='int')
            b_matrix, _ = sh_to_sf_matrix(sphere,
                                          find_order_from_nb_coeff(img_data),
                                          sh_basis, legacy=is_legacy)

            for idx in np.argwhere(np.sum(img_data, axis=-1)):
                idx = tuple(idx)
                directions, values, indices = get_maximas(img_data[idx],
                                                          sphere, b_matrix.T,
                                                          sf_threshold, 0)
                if values.shape[0] != 0:
                    n = min(npeaks, values.shape[0])
                    peak_dirs[idx][:n] = directions[:n]
                    peak_values[idx][:n] = values[:n]
                    peak_indices[idx][:n] = indices[:n]

            dg.peak_dirs = peak_dirs

        dg.peak_values = peak_values
        dg.peak_indices = peak_indices

        return dg


def get_theta(requested_theta, tracking_type):
    if requested_theta is not None:
        theta = requested_theta
    elif tracking_type == 'ptt':
        theta = 20
    elif tracking_type == 'prob':
        theta = 20
    elif tracking_type == 'eudx':
        theta = 60
    else:
        theta = 45
    return theta


def sample_distribution(dist, random_generator: np.random.Generator):
    """
    Parameters
    ----------
    dist: numpy.array
        The empirical distribution to sample from.
    random_generator: numpy Generator

    Return
    ------
    ind: int
        The index of the sampled element.
    """
    cdf = dist.cumsum()
    if cdf[-1] == 0:
        return None

    return cdf.searchsorted(random_generator.random() * cdf[-1])
