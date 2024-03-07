# -*- coding: utf-8 -*-
import logging
import os
import sys

import nibabel as nib
import numpy as np

from scilpy.image.volume_math import concatenate
from scilpy.io.image import load_img
from scilpy.io.utils import get_acq_parameters
from scilpy.reconst.mti import adjust_B1_map_intensities, smooth_B1_map, \
    process_contrast_map


def add_common_args_mti(p):
    """
    Defines arguments used in common for these scripts:
    - scil_mti_maps_MT.py
    - scil_mti_maps_ihMT.py
    """
    p.add_argument('--extended', action='store_true',
                   help='If set, outputs the folder Complementary_maps.')
    p.add_argument('--filtering', action='store_true',
                   help='Gaussian filtering to remove Gibbs ringing. '
                        'Not recommended.')

    a = p.add_argument_group(
        title='Acquisition parameters',
        description='Acquisition parameters required for MTsat and ihMTsat '
                    'calculation. \nThese are the excitation flip angles '
                    '(a_PD, a_T1), in DEGREES, and \nrepetition times (TR_PD, '
                    'TR_T1) of the PD and T1 images, in SECONDS. \nCan be '
                    'given through json files (--in_jsons) or directly '
                    '(--in_acq_parameters).')
    a1 = a.add_mutually_exclusive_group(required='--in_mtoff_t1' in sys.argv)
    a1.add_argument('--in_jsons', nargs=2,
                    metavar=('PD_json', 'T1_json'),
                    help='Path to MToff PD json file and MToff T1 json file, '
                         'in that order. \nThe acquisition parameters will be '
                         'extracted from these files. \nMust come from a '
                         'Philips acquisition, otherwise, use '
                         'in_acq_parameters.')
    a1.add_argument('--in_acq_parameters', nargs=4, type=float,
                    metavar=('PD flip angle', 'T1 flip angle',
                             'PD repetition time', 'T1 repetition time'),
                    help='Acquisition parameters in that order: flip angle of '
                         'mtoff_PD, \nflip angle of mtoff_T1, repetition time '
                         'of mtoff_PD, \nrepetition time of mtoff_T1')

    b = p.add_argument_group(title='B1 correction')
    b.add_argument('--in_B1_map',
                   help='Path to B1 coregister map to MT contrasts.')
    b.add_argument('--B1_correction_method',
                   choices=['empiric', 'model_based'], default='empiric',
                   help='Choice of B1 correction method. Choose between '
                        'empiric and model-based. \nNote that the model-based '
                        'method requires a B1 fitvalues file. \nBoth method '
                        'will only correct the saturation measures. '
                        '[%(default)s]')
    b.add_argument('--B1_fitvalues', nargs='+',
                   help='Path to B1 fitvalues files obtained externally. '
                        'Should be one .mat \nfile per input MT-on image, '
                        'given in this specific order: \npositive frequency '
                        'saturation, negative frequency saturation.')
    b.add_argument('--B1_nominal', default=100, type=float,
                   help='Nominal value for the B1 map. For Philips, should be '
                        '100. [%(default)s]')
    b.add_argument('--B1_smooth_dims', default=5, type=int,
                   help='Dimension of the squared window used for B1 '
                        'smoothing, in number of voxels. [%(default)s]')


def load_and_verify_mti(args, parser, input_maps_lists, extended_dir, affine,
                        contrast_names):
    """
    Common verifications and loading for both MT and ihMT scripts.

    Parameters
    ----------
    args: Namespace
    parser: Argparser
    input_maps_lists: list[list]
        A list of lists of inputs.
    extended_dir: str
        The folder for extended savings (with option args.extended).
    affine: np.ndarray
        A reference affine to save files.
    contrast_names: list
        A list of prefixes for each sub-list in input_maps_lists.

    Returns
    -------
    single_echo: bool
        True if the first list in input_maps_lists (i.e. the main echoes)
        contains only one file.
    flip_angles: list[float]
        The flip angles, in radian
    rep_times: list[float]
        The rep times, in ms.
    B1_map: np.ndarray
        The loaded map, with adjusted intensities, smoothed, corrected.
    contrast_maps: list[np.ndarray]
        One contrast map per string in contrast_names.
    """
    # Verify that there is the same number of --positive, --negative,
    # --in_mtoff_pd and --in_mtoff_t1
    for curr_map_list in input_maps_lists[1:]:
        if len(curr_map_list) != len(input_maps_lists[0]):
            parser.error('Not the same number of echoes per contrast')

    if len(input_maps_lists[0]) == 1:
        single_echo = True
    else:
        single_echo = False

    if args.in_B1_map and not args.in_mtoff_t1:
        logging.warning('No B1 correction was applied because no MTsat or '
                        'ihMTsat can be computed without the in_mtoff_t1.')

    if args.B1_correction_method == 'model_based' and not args.B1_fitvalues:
        parser.error('Fitvalues files must be given when choosing the '
                     'model-based B1 correction method. Please use '
                     '--B1_fitvalues.')

    # Set TR and FlipAngle parameters. Required with --in_mtoff_t1, in which
    # case one of --in_aqc_parameters or --in_jsons is set.
    rep_times, flip_angles = _parse_acquisition_parameters(args)

    # Fix issue from the presence of invalid value and division by zero
    np.seterr(divide='ignore', invalid='ignore')

    # Load B1 image
    B1_map, flip_angles = _prepare_B1_map(args, flip_angles, extended_dir,
                                          affine)

    # Define contrasts maps names
    if args.filtering:
        contrast_names = [curr_name + '_filter'
                          for curr_name in contrast_names]
    if single_echo:
        contrast_names = [curr_name + '_single_echo'
                          for curr_name in contrast_names]
    if args.out_prefix:
        contrast_names = [args.out_prefix + '_' + curr_name
                          for curr_name in contrast_names]

    # Compute contrasts maps
    contrast_maps = []
    for idx, curr_map in enumerate(input_maps_lists):
        input_images = []
        for image in curr_map:
            img, _ = load_img(image)
            input_images.append(img)
        merged_curr_map = concatenate(input_images, input_images[0])
        contrast_maps.append(process_contrast_map(merged_curr_map,
                                                  filtering=args.filtering,
                                                  single_echo=single_echo))
        if args.extended:
            nib.save(nib.Nifti1Image(contrast_maps[idx].astype(np.float32),
                                     affine),
                     os.path.join(extended_dir,
                                  contrast_names[idx] + '.nii.gz'))

    return single_echo, flip_angles, rep_times, B1_map, contrast_maps


def _parse_acquisition_parameters(args):
    """
    Parse the acquisition parameters from MTI, either from json files or
    directly inputed parameters.

    Parameters
    ----------
    args: Namespace

    Returns
    -------
    flip_angles: list[float]
        The flip angles, in radian
    rep_times: list[float]
        The rep times, in ms.
    """
    rep_times = None
    flip_angles = None
    if args.in_acq_parameters:
        flip_angles = np.asarray(args.in_acq_parameters[:2]) * np.pi / 180.
        rep_times = np.asarray(args.in_acq_parameters[2:]) * 1000
        if rep_times[0] > 10000 or rep_times[1] > 10000:
            logging.warning('Given repetition times do not seem to be given '
                            'in seconds. MTsat results might be affected.')
    elif args.in_jsons:
        rep_times = []
        flip_angles = []
        for curr_json in args.in_jsons:
            acq_parameter = get_acq_parameters(curr_json,
                                               ['RepetitionTime', 'FlipAngle'])
            if acq_parameter[0] > 10:
                logging.warning('Repetition time found in {} does not seem to '
                                'be given in seconds. MTsat and ihMTsat '
                                'results might be affected.'.format(curr_json))
            rep_times.append(acq_parameter[0] * 1000)  # convert to ms.
            flip_angles.append(np.deg2rad(acq_parameter[1]))
    return rep_times, flip_angles


def _prepare_B1_map(args, flip_angles, extended_dir, affine):
    """
    Prepare the B1 map for MTI B1+ inhomogeneity correction. Flip angles might
    also be affected.

    Parameters
    ----------
    args: Namespace
    flip_angles: list[float]
        The flip angles, in radian
    extended_dir: str
        The folder for extended savings (with option args.extended).
    affine: np.ndarray
        A reference affine to save files.

    Returns
    -------
    B1_map: np.ndarray
        The loaded map, with adjusted intensities, smoothed, corrected.
    flip_angles: list[float]
        The modified flip angles, in radian
    """
    B1_map = None
    if args.in_B1_map and args.in_mtoff_t1:
        B1_img = nib.load(args.in_B1_map)
        B1_map = B1_img.get_fdata(dtype=np.float32)
        B1_map = adjust_B1_map_intensities(B1_map, nominal=args.B1_nominal)
        B1_map = smooth_B1_map(B1_map, wdims=args.B1_smooth_dims)
        if args.B1_correction_method == 'model_based':
            # Apply the B1 map to the flip angles for model-based correction
            flip_angles[0] *= B1_map
            flip_angles[1] *= B1_map
        if args.extended:
            nib.save(nib.Nifti1Image(B1_map, affine),
                     os.path.join(extended_dir, "B1_map.nii.gz"))
    return B1_map, flip_angles
