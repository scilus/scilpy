# -*- coding: utf-8 -*-
import logging
from typing import Iterable

from dipy.core.sphere import HemiSphere, Sphere
from dipy.data import get_sphere
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter, PTTDirectionGetter)
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.stateful_tractogram import Origin, Space
from dipy.io.utils import create_tractogram_header, get_reference_info
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.tracking.streamlinespeed import compress_streamlines, length
import nibabel as nib
from nibabel.streamlines import TrkFile
from nibabel.streamlines.tractogram import LazyTractogram, TractogramItem
import numpy as np
import scipy.ndimage as ndi
from tqdm import tqdm

from scilpy.io.utils import (add_compression_arg, add_overwrite_arg,
                             add_sh_basis_args)
from scilpy.reconst.utils import (find_order_from_nb_coeff, get_maximas,
                                  is_data_peaks)


class TrackingDirection(list):
    """
    Tracking direction use as 3D cartesian direction (list(x,y,z))
    and has an index to work with discrete sphere.
    """

    def __init__(self, cartesian, index=None):
        super(TrackingDirection, self).__init__(cartesian)
        self.index = index


def add_mandatory_options_tracking(p, fodf_optional=False):
    """
    Args that are required in both scil_tracking_local and
    scil_tracking_local_dev scripts.
    """
    if fodf_optional:
        odf_group = p.add_mutually_exclusive_group()
        odf_group.add_argument('--in_odf', default=None,
                               help='File containing the orientation \n'
                               'diffusion function as spherical harmonics \n'
                               'file (.nii.gz). Ex: ODF or fODF. \n'
                               'If not provided, fODF info must be \n'
                               'specified in rap_policies.json.')
        odf_group.add_argument(
            '--rap_params', default=None,
            help='JSON file containing RAP parameters, mutually exclusive '
                 'with --in_odf.\nRequired for --rap_method switch.\n'
                 'Expected format:\n'
                 '{\n'
                 '  "methods": {\n'
                 '    "1": {"propagator": "ODF", "filename": str,\n'
                 '          "sh_basis": str, "algo": str,\n'
                 '          "theta": float, "step_size": float},\n'
                 '    "2": {"propagator": "ODF", "filename": str,\n'
                 '          "sh_basis": str, "algo": str,\n'
                 '          "theta": float, "step_size": float}\n'
                 '  }\n'
                 '}')
    else:
        p.add_argument('in_odf',
                       help='File containing the orientation diffusion '
                            'function \nas spherical harmonics file '
                            '(.nii.gz). \nEx: ODF or fODF.')
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
                         help='Spherical function relative threshold '
                              'within each voxel. [%(default)s]')
    global_sf_g = track_g.add_mutually_exclusive_group()
    global_sf_g.add_argument('--global_sf_rel_thr', metavar='FACTOR',
                             type=float, nargs='?', const=0.1, default=None,
                             help='Global SF relative threshold factor. '
                                  'If set, masks voxels where\nmaximum SF '
                                  'amplitude < FACTOR * global maximum SF '
                                  'amplitude. \nIf used without a value, '
                                  'default is [%(const)s].')
    global_sf_g.add_argument('--global_sf_abs_thr', metavar='ABS_THR',
                             type=float,
                             help='Global SF absolute threshold. '
                                  'If set, masks voxels where \n'
                                  'maximum SF amplitude < ABS_THR.')
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
    seed_sub_exclusive.add_argument(
        '--in_custom_seeds', type=str,
        help='Path to a file containing a list of custom seeding \n'
             'coordinates (.txt, .mat or .npy). They should be in \n'
             'voxel space. In the case of a text file, each line should \n'
             'contain a single seed, written in the format: [x, y, z].')


def add_out_options(p):
    """
    Options that are available in both scil_tracking_local and
    scil_tracking_local_dev scripts.
    """
    out_g = p.add_argument_group('Output options')
    msg = ("\nA rule of thumb is to set it to 0.1mm for deterministic \n"
           "streamlines and to 0.2mm for probabilistic streamlines.")
    add_compression_arg(out_g, additional_msg=msg)

    add_overwrite_arg(out_g)
    out_g.add_argument('--save_seeds', action='store_true',
                       help='If set, save the seeds used for the tracking \n '
                            'in the data_per_streamline property.\n'
                            'Hint: you can then use '
                            'scil_tractogram_seed_density_map.')
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
        out_tractogram, min_length, max_length, compress, save_seeds, verbose,
        space=Space.VOX, origin=Origin.NIFTI
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
    ref_img : nibabel.Nifti1Image or scilpy.io.stateful_image.StatefulImage
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
    space : Space
        Space in which the streamlines are generated.
    origin : Origin
        Origin in which the streamlines are generated.

    """
    voxel_size = np.array(ref_img.header.get_zooms()[:3])
    # If ref_img is a StatefulImage, we want to save relative to its
    # original on-disk orientation, not the internal (likely RAS) one.
    from scilpy.io.stateful_image import StatefulImage
    is_stateful = isinstance(ref_img, StatefulImage)

    if is_stateful:
        affine_mod = ref_img.affine.copy()
        affine_ori = ref_img._original_affine
        original_voxel_size = np.array(ref_img._original_voxel_sizes[:3])
    else:
        affine_mod = ref_img.affine.copy()
        affine_ori = ref_img.affine.copy()
        original_voxel_size = voxel_size

    def tracks_generator_wrapper():
        # If streamlines_generator is a callable, call it to get a new
        # generator. This allows re-iterating if LazyTractogram needs it.
        if callable(streamlines_generator):
            iterable = streamlines_generator()
        else:
            iterable = streamlines_generator

        miniters = int(total_nb_seeds / 100) \
            if total_nb_seeds >= 100 else 1
        for strl, seed in tqdm_if_verbose(iterable,
                                          verbose=verbose,
                                          total=total_nb_seeds,
                                          miniters=miniters,
                                          leave=False):
            # 1. Get to RASMM (physical world space) for filtering and
            # compression
            if space == Space.VOX:
                strl_rasmm = nib.affines.apply_affine(affine_mod, strl)
            elif space == Space.VOXMM:
                strl_rasmm = nib.affines.apply_affine(
                    affine_mod, strl / voxel_size)
            elif space == Space.RASMM:
                strl_rasmm = strl
            else:
                raise ValueError("Unknown space")

            strl_len = length(strl_rasmm)
            if (min_length <= strl_len <= max_length):
                # Prepare DPS for this streamline
                strl_dps = {}
                if save_seeds:
                    strl_dps['seeds'] = seed

                if compress:
                    strl_rasmm = compress_streamlines(strl_rasmm, compress)

                if tracts_format is TrkFile:
                    # TRK expects VOXMM relative to original orientation
                    strl_vox = nib.affines.apply_affine(
                        np.linalg.inv(affine_ori), strl_rasmm)
                    # Add half-voxel shift to go from scilpy center-origin
                    # to nibabel corner-origin in voxmm.
                    strl_to_save = (strl_vox + 0.5) * original_voxel_size
                else:
                    # TCK expects RASMM
                    strl_to_save = strl_rasmm

                yield TractogramItem(strl_to_save, strl_dps, {})

    tractogram = LazyTractogram.from_data_func(tracks_generator_wrapper)
    tractogram.affine_to_rasmm = np.eye(4)

    filetype = nib.streamlines.detect_format(out_tractogram)

    if is_stateful:
        reference = (ref_img._original_affine,
                     ref_img._original_dimensions[:3],
                     ref_img._original_voxel_sizes[:3],
                     "".join(ref_img._original_axcodes[:3]))
        header = create_tractogram_header(filetype, *reference)
    else:
        reference = get_reference_info(ref_img)
        header = create_tractogram_header(filetype, *reference)

    # Save
    if filetype is TrkFile:
        new_tractogram = nib.streamlines.TrkFile(tractogram, header)
    else:
        new_tractogram = nib.streamlines.TckFile(tractogram, header)

    nib.streamlines.save(new_tractogram, out_tractogram)


def get_direction_getter(img_data, algo, sphere, sub_sphere, theta, sh_basis,
                         voxel_size, sf_threshold, sh_to_pmf,
                         probe_length, probe_radius, probe_quality,
                         probe_count, support_exponent, is_legacy=True):
    """ Return the direction getter object.

    Parameters
    ----------
    img_data: np.ndarray
        ODF data (SH or Peaks).
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
        Map spherical harmonics to spherical function (pmf) before tracking
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
    sphere = HemiSphere.from_sphere(
        get_sphere(name=sphere)).subdivide(n=sub_sphere)

    # Theta depends on user choice and algorithm
    theta = get_theta(theta, algo)
    is_peaks = is_data_peaks(img_data)
    if is_peaks:
        logging.warning(
            'Input detected as peaks. Input should be fodf for '
            'det/prob/ptt, verify input just in case.')

    if algo in ['det', 'prob', 'ptt']:
        kwargs = {}
        if algo == 'ptt':
            dg_class = PTTDirectionGetter
            # Probe length and radius are in mm, convert to voxel units
            # since tracking is performed in voxel space (identity affine).
            kwargs = {'probe_length': probe_length / voxel_size,
                      'probe_radius': probe_radius / voxel_size,
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
            b_matrix, _ = sh_to_sf_matrix(
                sphere,
                sh_order_max=find_order_from_nb_coeff(img_data),
                basis_type=sh_basis, legacy=is_legacy)

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


def compute_max_sf_amplitude(data, sh_basis, is_legacy,
                             sphere_name='repulsion100', mask=None):
    """
    Compute the maximum SF amplitude for each voxel.
    Only computes SF for voxels where data is non-zero (or in mask) to save
    RAM.

    This information can be used to compute a global threshold for SF
    amplitude, which is often used to filter out spurious peaks in fODF.

    Parameters
    ----------
    data : np.ndarray
        ODF data (SH).
    sh_basis : str
        SH basis ('tournier07' or 'descoteaux07').
    is_legacy : bool
        Whether the SH basis is legacy.
    sphere_name : str or dipy.core.sphere.Sphere, optional
        Sphere name for SF conversion or Sphere object.
    mask : np.ndarray, optional
        Binary mask. If provided, only voxels in mask are computed.

    Returns
    -------
    max_sf : np.ndarray
        Maximum SF amplitude per voxel.
    """
    if mask is None:
        mask = np.any(data, axis=-1)

    order = find_order_from_nb_coeff(data)
    if isinstance(sphere_name, (Sphere,)):
        sphere = sphere_name
    else:
        sphere = get_sphere(name=sphere_name)

    b_matrix, _ = sh_to_sf_matrix(sphere, sh_order_max=order,
                                  basis_type=sh_basis, legacy=is_legacy)

    max_sf = np.zeros(data.shape[:-1], dtype=np.float32)
    if np.any(mask):
        # Vectorized SF computation for masked voxels
        sf = np.dot(data[mask], b_matrix)
        max_sf[mask] = np.max(sf, axis=-1)

    return max_sf


def compute_sf_threshold_mask(data, sphere_name='repulsion100',
                              relative_factor=None,
                              absolute_threshold=None,
                              sh_basis='descoteaux07',
                              is_legacy=True, postprocess_mask=True,
                              size_percentage=0.05):
    """
    Compute a binary mask based on a global SF amplitude threshold.

    In SF obtained from fODF, the amplitude of the lobes corresponds to the
    strength of the diffusion signal in those directions. Thresholding these
    amplitudes is a common practice to filter out spurious peaks.

    Parameters
    ----------
    data : np.ndarray
        ODF data (SH or Peaks).
    sphere_name : str or dipy.core.sphere.Sphere, optional
        Sphere name for SF conversion or Sphere object.
    relative_factor : float, optional
        Factor between 0 and 1. Threshold is factor * global_max_sf.
    absolute_threshold : float, optional
        Absolute threshold on SF amplitude.
    sh_basis : str, optional
        SH basis ('tournier07' or 'descoteaux07').
    is_legacy : bool, optional
        Whether the SH basis is legacy.
    postprocess_mask : bool, optional
        Whether to postprocess the mask to keep only the largest component.
    size_percentage : float, optional
        If postprocess_mask is True, percentage of the largest component size
        under which a hole will be filled.

    Returns
    -------
    mask : np.ndarray
        Binary mask.
    global_max : float
        Global maximum SF amplitude.
    threshold : float
        Computed threshold value.
    """
    if relative_factor is None and absolute_threshold is None:
        raise ValueError("Either relative_factor or absolute_threshold "
                         "must be provided.")

    is_peaks = is_data_peaks(data)
    if is_peaks:
        if data.ndim == 5:
            if data.shape[-1] != 3:
                raise ValueError("5D peaks input must have 3 "
                                 "as last dimension.")
            peaks = data
        elif data.ndim == 4:
            npeaks = data.shape[-1] // 3
            peaks = data.reshape(data.shape[:3] + (npeaks, 3))
        else:
            raise ValueError("Peaks input must be 4D or 5D.")

        norms = np.linalg.norm(peaks, axis=-1)
        # maximum amplitude/norm across peaks
        max_amp = np.max(norms, axis=-1)

        # Check for normalized peaks
        nonzero_norms = norms[norms > 0]
        if len(nonzero_norms) > 0 and \
                np.all(np.isclose(nonzero_norms, nonzero_norms[0])):
            logging.warning("All peaks have the same norm. They might be "
                            "already normalized.")
    else:
        max_amp = compute_max_sf_amplitude(data, sh_basis, is_legacy,
                                           sphere_name=sphere_name)

    global_max = np.max(max_amp)

    # Compute threshold. Use max if both are provided.
    threshold = 0
    if absolute_threshold is not None:
        threshold = absolute_threshold
    if relative_factor is not None:
        if relative_factor < 0 or relative_factor > 1:
            raise ValueError("relative_factor must be between 0 and 1.")
        threshold = max(threshold, relative_factor * global_max)

    if global_max == 0:
        mask = np.zeros(max_amp.shape, dtype=bool)
    else:
        mask = max_amp >= threshold

    if postprocess_mask and np.any(mask):
        # Postprocess to label all elements and count voxels for each label
        labels = ndi.label(mask)[0]
        label_counts = np.bincount(labels.ravel())

        # Guard against empty label_counts[1:]
        if len(label_counts) > 1:
            # Find the largest connected component (excluding background)
            # +1 to skip background
            largest_label = np.argmax(label_counts[1:]) + 1
            largest_component_size = label_counts[largest_label]

            # Create a mask for the largest connected component
            mask = labels == largest_label
            inverted_mask = ~mask

            # Remove isolated voxels in the inverted mask (holes in main mask)
            labels_inverted = ndi.label(inverted_mask)[0]
            label_counts_inverted = np.bincount(labels_inverted.ravel())

            # Fill holes smaller than X% of the largest component size
            hole_threshold = size_percentage * largest_component_size
            for label, count in enumerate(label_counts_inverted):
                if label == 0:
                    continue  # Skip background
                if count < hole_threshold:
                    mask[labels_inverted == label] = True

    return mask, global_max, threshold


def get_global_sf_threshold_mask(data, args, sh_basis, is_legacy):
    """
    Wrapper for compute_sf_threshold_mask to compute the global SF
    threshold mask and log information.

     The global SF threshold can be set as a relative factor of the global
     maximum SF amplitude, or as an absolute threshold. The relative factor is
     often set between 0.1 and 0.2, but it can depend on the data and the
     SH basis used. The absolute threshold can be estimated from the
     mean/median maximum fODF in the ventricles, computed with
     scil_fodf_max_in_ventricles.

     Note that this estimation is not perfect as it depends on the accuracy of
     the ventricle mask and on the presence of noise/artifacts in the data.

    Parameters
    ----------
    data : np.ndarray
        ODF data (SH or Peaks).
    args : argparse.Namespace
        Arguments from the CLI. Must contain sphere, global_sf_rel_thr,
        and global_sf_abs_thr.
    sh_basis : str
        SH basis.
    is_legacy : bool
        Whether the SH basis is legacy.

    Returns
    -------
    sf_mask : np.ndarray
        Binary mask.
    """
    sf_mask, global_max, threshold = compute_sf_threshold_mask(
        data, sphere_name=args.sphere,
        relative_factor=args.global_sf_rel_thr,
        absolute_threshold=args.global_sf_abs_thr, sh_basis=sh_basis,
        is_legacy=is_legacy)
    logging.info("Global SF threshold mask: Global Max SF amplitude: "
                 "{:.4f}".format(global_max))
    if args.global_sf_rel_thr is not None:
        logging.info("Global SF threshold mask: Computed threshold: "
                     "{:.4f} (Factor: {})"
                     .format(threshold, args.global_sf_rel_thr))
    else:
        logging.info("Global SF threshold mask: Absolute threshold: "
                     "{:.4f}".format(args.global_sf_abs_thr))
    return sf_mask
