# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import multiprocessing
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np
from dipy.data import SPHERE_FILES
from dipy.io.utils import is_header_compatible
from fury import window
from PIL import Image
from scipy.io import loadmat
import six

from scilpy.gradients.bvec_bval_tools import DEFAULT_B0_THRESHOLD
from scilpy.utils.filenames import split_name_with_nii

eddy_options = ["mb", "mb_offs", "slspec", "mporder", "s2v_lambda", "field",
                "field_mat", "flm", "slm", "fwhm", "niter", "s2v_niter",
                "cnr_maps", "residuals", "fep", "interp", "s2v_interp",
                "resamp", "nvoxhp", "ff", "ol_nstd", "ol_nvox", "ol_type",
                "ol_pos", "ol_sqr", "dont_sep_offs_move", "dont_peas"]

topup_options = ['out', 'fout', 'iout', 'logout', 'warpres', 'subsamp', 'fwhm',
                 'config', 'miter', 'lambda', 'ssqlambda', 'regmod', 'estmov',
                 "minmet", 'splineorder', 'numprec', 'interp', 'scale',
                 'regrid']

axis_name_choices = ["axial", "coronal", "sagittal"]


def get_acq_parameters(json_path, args_list):
    """
    Function to extract acquisition parameters from json file.

    Parameters
    ----------
    json_path   Path to the json file
    args_list   List of keys corresponding to parameters

    Returns
    ----------
    Returns a list of values matching the list of keys.
    """
    with open(json_path) as f:
        data = json.load(f)

    acq_parameters = []
    for parameter in args_list:
        acq_parameters.append(data[parameter])
    return acq_parameters


def redirect_stdout_c():
    sys.stdout.flush()
    newstdout = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    sys.stdout = os.fdopen(newstdout, 'w')


def link_bundles_and_reference(parser, args, input_tractogram_list):
    """
    Associate the bundle to their reference (if they require a reference).
    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser as created by argparse.
    args: argparse namespace
        Args as created by argparse.
    input_tractogram_list: list
        List of tractogram paths.
    Returns
    -------
    list: List of tuples, each matching one tractogram to a reference file.
    """
    bundles_references_tuple = []
    for bundle_filename in input_tractogram_list:
        _, ext = os.path.splitext(bundle_filename)
        if ext == '.trk':
            if args.reference is None:
                bundles_references_tuple.append(
                    (bundle_filename, bundle_filename))
            else:
                bundles_references_tuple.append(
                    (bundle_filename, args.reference))
        elif ext in ['.tck', '.fib', '.vtk', '.dpy']:
            if args.reference is None:
                parser.error('--reference is required for this file format '
                             '{}.'.format(bundle_filename))
            else:
                bundles_references_tuple.append(
                    (bundle_filename, args.reference))
    return bundles_references_tuple


def check_tracts_same_format(parser, filename_list):
    _, ref_ext = os.path.splitext(filename_list[0])

    for filename in filename_list[1:]:
        if isinstance(filename, six.string_types) and \
                not os.path.splitext(filename)[1] == ref_ext:
            parser.error('All tracts file must use the same format.')


def assert_gradients_filenames_valid(parser, filename_list, input_is_fsl):
    """
    Validate if gradients filenames follow BIDS or MRtrix convention

    Parameters
    ----------
    parser: parser
        Parser.
    filename_list: list
        list of gradient paths.
    input_is_fsl: bool
        Whether the input is in FSL format or MRtrix format.

    """

    valid_fsl_extensions = ['.bval', '.bvec']
    valid_mrtrix_extension = '.b'

    if isinstance(filename_list, str):
        filename_list = [filename_list]

    if input_is_fsl:
        if len(filename_list) == 2:
            filename_1 = filename_list[0]
            filename_2 = filename_list[1]
            basename_1, ext_1 = os.path.splitext(filename_1)
            basename_2, ext_2 = os.path.splitext(filename_2)

            if ext_1 == '' or ext_2 == '':
                parser.error('fsl gradients filenames must have extensions: '
                             '.bval and .bvec.')

            if basename_1 == basename_2:
                curr_extensions = [ext_1, ext_2]
                curr_extensions.sort()
                if curr_extensions != valid_fsl_extensions:
                    parser.error('Your extensions ({}) doesn\'t follow BIDS '
                                 'convention.'.format(curr_extensions))
            else:
                parser.error('fsl gradients filenames must have the same '
                             'basename.')
        else:
            parser.error('You should have two files for fsl format.')

    else:
        if len(filename_list) == 1:
            curr_filename = filename_list[0]
            basename, ext = os.path.splitext(curr_filename)
            if basename == '' or ext != valid_mrtrix_extension:
                parser.error('Basename: {} and extension {} are not '
                             'valid for mrtrix format.'.format(basename, ext))
        else:
            parser.error('You should have one file for mrtrix format.')


def add_json_args(parser):
    g1 = parser.add_argument_group(title='Json options')
    g1.add_argument('--indent',
                    type=int, default=2,
                    help='Indent for json pretty print.')
    g1.add_argument('--sort_keys',
                    action='store_true',
                    help='Sort keys in output json.')


def add_processes_arg(parser):
    parser.add_argument('--processes', dest='nbr_processes',
                        metavar='NBR', type=int, default=1,
                        help='Number of sub-processes to start. \n'
                             'Default: [%(default)s]')


def add_reference_arg(parser, arg_name=None):
    if arg_name:
        parser.add_argument('--' + arg_name + '_ref',
                            help='Reference anatomy for {} (if tck/vtk/fib/dpy'
                                 ') file\n'
                                 'support (.nii or .nii.gz).'.format(arg_name))
    else:
        parser.add_argument('--reference',
                            help='Reference anatomy for tck/vtk/fib/dpy file\n'
                                 'support (.nii or .nii.gz).')


def add_sphere_arg(parser, symmetric_only=False, default='repulsion724'):
    spheres = sorted(SPHERE_FILES.keys())
    if symmetric_only:
        spheres = [s for s in spheres if 'symmetric' in s]
        if 'symmetric' not in default:
            raise ValueError("Default cannot be {} if you only accept "
                             "symmetric spheres.".format(default))

    parser.add_argument('--sphere', choices=spheres,
                        default=default,
                        help='Dipy sphere; set of possible directions.\n'
                             'Default: [%(default)s]')


def add_overwrite_arg(parser):
    parser.add_argument(
        '-f', dest='overwrite', action='store_true',
        help='Force overwriting of the output files.')


def add_tolerance_arg(parser):
    parser.add_argument(
        '--tolerance', type=int, default=20, metavar='tol',
        help='The tolerated gap between the b-values to extract and the '
             'current b-value.\n'
             '[Default: %(default)s]\n'
             '* Note. We would expect to find at least one b-value in the \n'
             '  range [0, tolerance]. To skip this check, use '
             '--skip_b0_check.')


def add_b0_thresh_arg(parser):
    parser.add_argument(
        '--b0_threshold', type=float, default=DEFAULT_B0_THRESHOLD,
        metavar='thr',
        help='Threshold under which b-values are considered to be b0s.\n'
             '[Default: %(default)s] \n'
             '* Note. We would expect to find at least one b-value in the \n'
             '  range [0, b0_threshold]. To skip this check, use '
             '--skip_b0_check.')


def add_skip_b0_check_arg(parser, will_overwrite_with_min,
                          b0_tol_name='--b0_threshold'):
    """
    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    will_overwrite_with_min: bool
        If true, the help message will explain that b0_threshold could be
        overwritten.
    b0_tol_name: str
        Name of the argparse parameter acting as b0_threshold. Should probably
        be either '--b0_threshold' or '--tolerance'.
    """
    msg = ('By default, we supervise that at least one b0 exists in your '
           'data\n'
           '(i.e. b-values below the default {}). Use this option to \n'
           'allow continuing even if the minimum b-value is suspiciously '
           'high.\n'.format(b0_tol_name))
    if will_overwrite_with_min:
        msg += ('If no b-value is found below the threshold, the script will '
                'continue \nwith your minimal b-value as new {}.\n'
                .format(b0_tol_name))
    msg += 'Use with care, and only if you understand your data.'

    parser.add_argument(
        '--skip_b0_check', action='store_true', help=msg)


def add_verbose_arg(parser):
    parser.add_argument('-v', default="WARNING", const='INFO', nargs='?',
                        choices=['DEBUG', 'INFO', 'WARNING'], dest='verbose',
                        help='Produces verbose output depending on '
                             'the provided level. \nDefault level is warning, '
                             'default when using -v is info.')


def add_bbox_arg(parser):
    parser.add_argument('--no_bbox_check', dest='bbox_check',
                        action='store_false',
                        help='Activate to ignore validity of the bounding '
                             'box during loading / saving of \n'
                             'tractograms (ignores the presence of invalid '
                             'streamlines).')


def add_sh_basis_args(parser, mandatory=False, input_output=False):
    """
    Add spherical harmonics (SH) bases argument. For more information about
    the bases, see https://docs.dipy.org/stable/theory/sh_basis.html.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    mandatory: bool
        Whether this argument is mandatory.
    input_output: bool
        Whether this argument should expect both input and output bases or not.
        If set, the sh_basis argument will expect first the input basis,
        followed by the output basis.
    """
    if input_output:
        nargs = 2
        def_val = ['descoteaux07_legacy', 'tournier07']
        input_output_msg = ('\nBoth the input and output bases are '
                            'required, in that order.')
    else:
        nargs = 1
        def_val = ['descoteaux07_legacy']
        input_output_msg = ''

    choices = ['descoteaux07', 'tournier07', 'descoteaux07_legacy',
               'tournier07_legacy']
    help_msg = ("Spherical harmonics basis used for the SH coefficients. "
                "{}\n"
                "Must be either descoteaux07', 'tournier07', \n"
                "'descoteaux07_legacy' or 'tournier07_legacy' [%(default)s]:\n"
                "    'descoteaux07'       : SH basis from the Descoteaux "
                "et al.\n"
                "                           MRM 2007 paper\n"
                "    'tournier07'         : SH basis from the new "
                "Tournier et al.\n"
                "                           NeuroImage 2019 paper, as in "
                "MRtrix 3.\n"
                "    'descoteaux07_legacy': SH basis from the legacy Dipy "
                "implementation\n"
                "                           of the Descoteaux et al. MRM 2007 "
                "paper\n"
                "    'tournier07_legacy'  : SH basis from the legacy "
                "Tournier et al.\n"
                "                           NeuroImage 2007 paper."
                .format(input_output_msg))

    if mandatory:
        arg_name = 'sh_basis'
    else:
        arg_name = '--sh_basis'

    parser.add_argument(arg_name, nargs=nargs,
                        choices=choices, default=def_val,
                        help=help_msg)


def parse_sh_basis_arg(args):
    """
    Parser the input from args.sh_basis. If two SH bases are given,
    both input/output sh_basis and is_legacy are returned.

    Parameters
    ----------
    args : ArgumentParser.parse_args
        ArgumentParser.parse_args from a script.

    Returns
    -------
    if args.sh_basis is a list of one string:
        sh_basis : string
            Spherical harmonic basis name.
        is_legacy : bool
            Whether the SH basis is in its legacy form.
    else: (args:sh_basis is a list of two strings)
        Returns a Tuple of 4 values:
        (sh_basis_in, is_legacy_in, sh_basis_out, is_legacy_out)
    """
    sh_basis_name = args.sh_basis[0]
    sh_basis = 'descoteaux07' if 'descoteaux07' in sh_basis_name \
        else 'tournier07'
    is_legacy = 'legacy' in sh_basis_name
    if len(args.sh_basis) == 2:
        sh_basis_name = args.sh_basis[1]
        out_sh_basis = 'descoteaux07' if 'descoteaux07' in sh_basis_name \
            else 'tournier07'
        is_out_legacy = 'legacy' in sh_basis_name
        return sh_basis, is_legacy, out_sh_basis, is_out_legacy
    else:
        return sh_basis, is_legacy


def add_nifti_screenshot_default_args(
        parser, slice_ids_mandatory=True, transparency_mask_mandatory=True
):
    _mask_prefix = "" if transparency_mask_mandatory else "--"

    _slice_ids_prefix, _slice_ids_help = "", "Slice indices to screenshot."
    _output_help = "Name of the output image (e.g. img.jpg, img.png)."
    if not slice_ids_mandatory:
        _slice_ids_prefix = "--"
        _slice_ids_help += " If None are supplied, all slices inside " \
                           "the transparency mask are selected."
        _output_help = "Name of the output image(s). If multiple slices are " \
                       "provided (or none), their index will be append to " \
                       "the name (e.g. volume.jpg, volume.png becomes " \
                       "volume_slice_0.jpg, volume_slice_0.png)."

    # Positional arguments
    parser.add_argument(
        "in_volume", help="Input 3D Nifti file (.nii/.nii.gz).")
    parser.add_argument("out_fname", help=_output_help)

    # Variable arguments
    parser.add_argument(
        f"{_mask_prefix}in_transparency_mask",
        help="Transparency mask 3D Nifti image (.nii/.nii.gz).")
    parser.add_argument(
        f"{_slice_ids_prefix}slice_ids", nargs="+", type=int,
        help=_slice_ids_help)

    # Optional arguments
    parser.add_argument(
        "--volume_cmap_name", default=None,
        help="Colormap name for the volume image data. [%(default)s]")
    parser.add_argument(
        "--axis_name", default="axial", type=str, choices=axis_name_choices,
        help="Name of the axis to visualize. [%(default)s]")
    parser.add_argument(
        "--win_dims", nargs=2, metavar=("WIDTH", "HEIGHT"), default=(768, 768),
        type=int, help="The dimensions for the vtk window. [%(default)s]")
    parser.add_argument(
        "--display_slice_number", action="store_true",
        help="If true, displays the slice number in the upper left corner."
    )
    parser.add_argument(
        "--display_lr", action="store_true",
        help="If true, add left and right annotations to the images."
    )


def add_nifti_screenshot_overlays_args(
        parser, labelmap_overlay=True, mask_overlay=True,
        transparency_is_overlay=False
):
    if labelmap_overlay:
        parser.add_argument(
            "--in_labelmap", help="Labelmap 3D Nifti image (.nii/.nii.gz).")
        parser.add_argument(
            "--labelmap_cmap_name", default="viridis",
            help="Colormap name for the labelmap image data. [%(default)s]")
        parser.add_argument(
            "--labelmap_alpha", type=ranged_type(float, 0., 1.), default=0.7,
            help="Opacity value for the labelmap overlay. [%(default)s].")

    if mask_overlay:
        if not transparency_is_overlay:
            parser.add_argument(
                "--in_masks", nargs="+",
                help="Mask 3D Nifti image (.nii/.nii.gz).")

        parser.add_argument(
            "--masks_colors", nargs="+", metavar="R G B",
            type=ranged_type(int, 0, 255), default=None,
            help="Colors for the mask overlay or contour")
        parser.add_argument(
            "--masks_as_contours", action='store_true',
            help="Create contours from masks instead "
                 "of overlays. [%(default)s].")
        parser.add_argument(
            "--masks_alpha", type=ranged_type(float, 0., 1.), default=0.7,
            help="Opacity value for the masks overlays. [%(default)s].")


def validate_nbr_processes(parser, args):
    """
    Check if the passed number of processes arg is valid.

    Valid values are considered to be in the [0, CPU count] range:
        - Raises a parser.error if an invalid value is provided.
        - Returns the maximum number of cores retrieved if no value (or a value
          of 0) is provided.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser as created by argparse.
    args: argparse namespace
        Args as created by argparse.

    Returns
    -------
    nbr_cpu: int
        The number of CPU to be used.
    """

    if args.nbr_processes:
        nbr_cpu = args.nbr_processes
    else:
        nbr_cpu = multiprocessing.cpu_count()

    if nbr_cpu < 0:
        parser.error('Number of processes must be > 0.')
    elif nbr_cpu > multiprocessing.cpu_count():
        parser.error('Max number of processes is {}. Got {}.'.format(
            multiprocessing.cpu_count(), nbr_cpu))

    return nbr_cpu


def validate_sh_basis_choice(sh_basis):
    """
    Check if the passed sh_basis arg to a fct is right.

    Parameters
    ----------
    sh_basis: str
        Either 'descoteaux08' or 'tournier07'

    Raises
    ------
    ValueError
        If sh_basis is not one of 'descoteaux07' or 'tournier07'
    """
    if not (sh_basis == 'descoteaux07' or sh_basis == 'tournier07'):
        raise ValueError("sh_basis should be either 'descoteaux07' or"
                         "'tournier07'.")


def verify_compression_th(compress_th):
    """
    Verify that the compression threshold is between 0.001 and 1. Else,
    produce a warning.

    Parameters
    -----------
    compress_th: float, the compression threshold.
    """
    if compress_th:
        if compress_th < 0.001 or compress_th > 1:
            logging.warning(
                'You are using an error rate of {}.\nWe recommend setting it '
                'between 0.001 and 1.\n0.001 will do almost nothing to the '
                'tracts while 1 will higly compress/linearize the tracts.'
                .format(compress_th))


def assert_inputs_exist(parser, required, optional=None):
    """
    Assert that all inputs exist. If not, print parser's usage and exit.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    required: string or list of paths
        Required paths to be checked.
    optional: string or list of paths
        Optional paths to be checked.
    """

    def check(path):
        if not os.path.isfile(path):
            parser.error('Input file {} does not exist'.format(path))

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for required_file in required:
        check(required_file)
    for optional_file in optional or []:
        if optional_file is not None:
            check(optional_file)


def assert_outputs_exist(parser, args, required, optional=None,
                         check_dir_exists=True):
    """
    Assert that all outputs don't exist or that if they exist, -f was used.
    If not, print parser's usage and exit.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    args: argparse namespace
        Argument list.
    required: string or list of paths to files
        Required paths to be checked.
    optional: string or list of paths to files
        Optional paths to be checked.
    check_dir_exists: bool
        Test if output directory exists.
    """

    def check(path):
        if os.path.isfile(path) and not args.overwrite:
            parser.error('Output file {} exists. Use -f to force '
                         'overwriting'.format(path))

        if check_dir_exists:
            path_dir = os.path.dirname(path)
            if path_dir and not os.path.isdir(path_dir):
                parser.error('Directory {}/ \n for a given output file '
                             'does not exists.'.format(path_dir))

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for required_file in required:
        check(required_file)
    for optional_file in optional or []:
        if optional_file:
            check(optional_file)


def assert_output_dirs_exist_and_empty(parser, args, required,
                                       optional=None, create_dir=True):
    """
    Assert that all output directories exist.
    If not, print parser's usage and exit.
    If exists and not empty, and -f used, delete dirs.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    args: argparse namespace
        Argument list.
    required: string or list of paths to files
        Required paths to be checked.
    optional: string or list of paths to files
        Optional paths to be checked.
    create_dir: bool
        If true, create the directory if it does not exist.
    """

    def check(path):
        if not os.path.isdir(path):
            if not create_dir:
                parser.error(
                    'Output directory {} doesn\'t exist.'.format(path))
            else:
                os.makedirs(path, exist_ok=True)
        if os.listdir(path):
            if not args.overwrite:
                parser.error(
                    'Output directory {} isn\'t empty and some files could be '
                    'overwritten or even deleted. Use -f option if you want '
                    'to continue.'.format(path))
            else:
                for the_file in os.listdir(path):
                    file_path = os.path.join(path, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)

    if isinstance(required, str):
        required = [required]
    if isinstance(optional, str):
        optional = [optional]

    for cur_dir in required:
        check(cur_dir)
    for opt_dir in optional or []:
        if opt_dir:
            check(opt_dir)


def assert_overlay_colors(colors, overlays, parser):
    if colors is None:
        return

    if len(colors) % 3 != 0:
        parser.error(
            "Masks colors must be tuples of 3 ints in the range [0, 255]")

    if len(colors) > 3 and len(colors) // 3 != len(overlays):
        parser.error(f"Bad number of colors supplied for overlays "
                     f"({len(overlays)}). Either provide no color, a "
                     f"single mask color or as many colors as there is masks")


def assert_roi_radii_format(parser):
    """
    Verifies the format of the inputed roi radii.

    Parameters
    ----------
    parser: argument parser
        Will raise an error if the --roi_radii format is wrong.

    Returns
    -------
    roi_radii: int or numpy array
        Roi radii as a scalar or an array of size (3,).
    """
    args = parser.parse_args()
    if len(args.roi_radii) == 1:
        roi_radii = args.roi_radii[0]
    elif len(args.roi_radii) == 3:
        roi_radii = args.roi_radii
    else:
        parser.error('Wrong size for --roi_radii, can only be a scalar' +
                     'or an array of size (3,)')
    return roi_radii


def assert_headers_compatible(parser, required, optional=None, reference=None):
    """
    Verifies the compatibility between the first item in list_files
    and the remaining files in list.

    Arguments
    ---------
    parser: argument parser
        Will raise an error if a file is not compatible.
    required: List[str]
        List of files to test
    optional: List[str or None]
        List of files. May contain None, they will be discarted.
    reference: str
        Reference for any .tck passed in `list_files`
    """
    all_valid = True

    # Format required and optional to lists if a single filename was sent.
    if isinstance(required, str):
        required = [required]
    if optional is None:
        optional = []
    elif isinstance(optional, str):
        optional = [optional]
    else:
        optional = [f for f in optional if f is not None]
    list_files = required + optional

    if reference is not None:
        list_files.append(reference)

    if len(list_files) <= 1:
        return

    # Gather "headers" for all files to compare against each other later
    headers = []
    for filepath in list_files:
        _, in_extension = split_name_with_nii(filepath)
        if in_extension in ['.trk', '.nii', '.nii.gz']:
            headers.append(filepath)
        elif in_extension == '.tck':
            if reference is None:
                parser.error(
                    '{} must be provided with a reference.'.format(filepath))
        else:
            parser.error('{} does not have a supported extension.'.format(
                filepath))

    # Verify again that we have more than one header (ex, if not all tck files)
    if len(headers) <= 1:
        return

    for curr in headers[1:]:
        if not is_header_compatible(headers[0], curr):
            # Not raising error now. Allows to show all errors.
            logging.error('ERROR:\"{}\" and \"{}\" do not have compatible '
                          'headers.'.format(headers[0], curr))
            all_valid = False

    if not all_valid:
        parser.error('Not all input files have compatible headers.')


def read_info_from_mb_bdo(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    geometry = root.attrib['type']
    center_tag = root.find('origin')
    flip = [-1, -1, 1]
    center = [flip[0] * float(center_tag.attrib['x'].replace(',', '.')),
              flip[1] * float(center_tag.attrib['y'].replace(',', '.')),
              flip[2] * float(center_tag.attrib['z'].replace(',', '.'))]
    row_list = tree.iter('Row')
    radius = [None, None, None]
    for i, row in enumerate(row_list):
        for j in range(0, 3):
            if j == i:
                key = 'col' + str(j + 1)
                radius[i] = float(row.attrib[key].replace(',', '.'))
            else:
                key = 'col' + str(j + 1)
                value = float(row.attrib[key].replace(',', '.'))
                if abs(value) > 0.01:
                    raise ValueError('Does not support rotation, for now \n'
                                     'only SO aligned on the X,Y,Z axis are '
                                     'supported.')
    radius = np.asarray(radius, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    return geometry, radius, center


def load_matrix_in_any_format(filepath):
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        data = np.loadtxt(filepath)
    elif ext == '.npy':
        data = np.load(filepath)
    elif ext == '.mat':
        # .mat are actually dictionnary. This function support .mat from
        # antsRegistration that encode a 4x4 transformation matrix.
        transfo_dict = loadmat(filepath)
        lps2ras = np.diag([-1, -1, 1])
        transfo_key = 'AffineTransform_double_3_3'
        if transfo_key not in transfo_dict:
            transfo_key = 'AffineTransform_float_3_3'

        rot = transfo_dict[transfo_key][0:9].reshape((3, 3))
        trans = transfo_dict[transfo_key][9:12]
        offset = transfo_dict['fixed']
        r_trans = (np.dot(rot, offset) - offset - trans).T * [1, 1, -1]

        data = np.eye(4)
        data[0:3, 3] = r_trans
        data[:3, :3] = np.dot(np.dot(lps2ras, rot), lps2ras)
    else:
        raise ValueError('Extension {} is not supported'.format(ext))

    return data


def save_matrix_in_any_format(filepath, output_data):
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        np.savetxt(filepath, output_data)
    elif ext == '.npy':
        np.save(filepath, output_data)
    elif ext == '':
        np.save('{}.npy'.format(filepath), output_data)
    else:
        raise ValueError('Extension {} is not supported'.format(ext))


def assert_fsl_options_exist(parser, options_args, command):
    """
    Assert that all options for topup or eddy exist.
    If not, print parser's usage and exit.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    options_args: string
        Options for fsl command
    command: string
        Command used (eddy or topup).
    """
    if command == 'eddy':
        fsl_options = eddy_options
    elif command == 'topup':
        fsl_options = topup_options
    else:
        parser.error('{} command is not supported as fsl '
                     'command.'.format(command))

    options = re.split(r'[ =\s]\s*', options_args)
    res = [i for i in options if "--" in i]
    res = list(map(lambda x: x.replace('--', ''), res))

    for nOption in res:
        if nOption not in fsl_options:
            parser.error('--{} is not a valid option for '
                         '{} command.'.format(nOption, command))


def parser_color_type(arg):
    """
    Validate that a color component is between RBG values, else return an error
    From https://stackoverflow.com/a/55410582
    """

    MIN_VAL = 0
    MAX_VAL = 255
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Color component must be a floating "
                                         "point number")
    if f < MIN_VAL or f > MAX_VAL:
        raise argparse.ArgumentTypeError(
            "Argument must be < " + str(MAX_VAL) + "and > " + str(MIN_VAL))
    return f


def snapshot(scene, filename, **kwargs):
    """ Wrapper around fury.window.snapshot
    For some reason, fury.window.snapshot flips the image vertically.
    This image unflips the image and then saves it.
    """
    out = window.snapshot(scene, **kwargs)
    image = Image.fromarray(out[::-1])
    image.save(filename)


def ranged_type(value_type, min_value, max_value):
    """Return a function handle of an argument type function for ArgumentParser
    checking a range: `min_value` <= arg <= `max_value`.

    Parameters
    ----------
    value_type : Type
        Value-type to convert the argument.
    min_value : scalar
        Minimum acceptable argument value.
    max_value : scalar
       Maximum acceptable argument value.

    Returns
    -------
    Function handle of an argument type function for ArgumentParser.

    Usage
    -----
        ranged_type(float, 0.0, 1.0)
    """

    def range_checker(arg: str):
        try:
            f = value_type(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"must be a valid {value_type}")
        if f < min_value or f > max_value:
            raise argparse.ArgumentTypeError(
                f"must be within [{min_value}, {max_value}]")
        return f

    # Return handle to checking function
    return range_checker


def get_default_screenshotting_data(args):
    volume_img = nib.load(args.in_volume)

    transparency_mask_img = None
    if args.in_transparency_mask:
        transparency_mask_img = nib.load(args.in_transparency_mask)

    labelmap_img = None
    if args.in_labelmap:
        labelmap_img = nib.load(args.in_labelmap)

    mask_imgs, masks_colors = None, None
    if args.in_masks:
        mask_imgs = [nib.load(f) for f in args.in_masks]

        if args.masks_colors is not None:
            if len(args.masks_colors) == 3:
                masks_colors = np.repeat(
                    [args.masks_colors], len(args.in_masks), axis=0)
            elif len(args.masks_colors) // 3 == len(args.in_masks):
                masks_colors = np.array(args.masks_colors).reshape((-1, 3))

            masks_colors = masks_colors / 255.

    return volume_img, \
        transparency_mask_img, \
        labelmap_img, \
        mask_imgs, \
        masks_colors
