# -*- coding: utf-8 -*-

import os
import multiprocessing
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.io import loadmat
import six

from scilpy.utils.bvec_bval_tools import DEFAULT_B0_THRESHOLD


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


def assert_gradients_filenames_valid(parser, filename_list, gradient_format):
    """
    Validate if gradients filenames follow BIDS or MRtrix convention

    Parameters
    ----------
    parser: parser
        Parser.
    filename_list: list
        list of gradient paths.
    gradient_format : str
        Can be either fsl or mrtrix.

    Returns
    -------
    """

    valid_fsl_extensions = ['.bval', '.bvec']
    valid_mrtrix_extension = '.b'

    if isinstance(filename_list, str):
        filename_list = [filename_list]

    if gradient_format == 'fsl':
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

    elif gradient_format == 'mrtrix':
        if len(filename_list) == 1:
            curr_filename = filename_list[0]
            basename, ext = os.path.splitext(curr_filename)
            if basename == '' or ext != valid_mrtrix_extension:
                parser.error('Basename: {} and extension {} are not '
                             'valid for mrtrix format.'.format(basename, ext))
        else:
            parser.error('You should have one file for mrtrix format.')
    else:
        parser.error('Gradient file format should be either fsl or mrtrix.')


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
        parser.add_argument('--'+arg_name+'_ref',
                            help='Reference anatomy for {} (if tck/vtk/fib/dpy'
                                 ') file\n'
                                 'support (.nii or .nii.gz).'.format(arg_name))
    else:
        parser.add_argument('--reference',
                            help='Reference anatomy for tck/vtk/fib/dpy file\n'
                                 'support (.nii or .nii.gz).')


def add_overwrite_arg(parser):
    parser.add_argument(
        '-f', dest='overwrite', action='store_true',
        help='Force overwriting of the output files.')


def add_force_b0_arg(parser):
    parser.add_argument('--force_b0_threshold', action='store_true',
                        help='If set, the script will continue even if the '
                             'minimum bvalue is suspiciously high ( > {})'
                        .format(DEFAULT_B0_THRESHOLD))


def add_verbose_arg(parser):
    parser.add_argument('-v', action='store_true', dest='verbose',
                        help='If set, produces verbose output.')


def add_sh_basis_args(parser, mandatory=False):
    """Add spherical harmonics (SH) bases argument.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser.
    mandatory: bool
        Whether this argument is mandatory.
    """
    choices = ['descoteaux07', 'tournier07']
    def_val = 'descoteaux07'
    help_msg = 'Spherical harmonics basis used for the SH coefficients.\nMust ' +\
               'be either \'descoteaux07\' or \'tournier07\' [%(default)s]:\n' +\
               '    \'descoteaux07\': SH basis from the Descoteaux et al.\n' +\
               '                      MRM 2007 paper\n' +\
               '    \'tournier07\'  : SH basis from the Tournier et al.\n' +\
               '                      NeuroImage 2007 paper.'

    if mandatory:
        arg_name = 'sh_basis'
    else:
        arg_name = '--sh_basis'

    parser.add_argument(arg_name,
                        choices=choices, default=def_val,
                        help=help_msg)


def validate_nbr_processes(parser, args, default_nbr_cpu=None):
    """ Check if the passed number of processes arg is valid.
    If not valid (0 < nbr_cpu_to_use <= cpu_count), raise parser.error.

    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser as created by argparse.
    args: argparse namespace
        Args as created by argparse.
    default_nbr_cpu: int (or None)
        Number of cpu to use, default is cpu_count (all).

    Results
    ------
    nbr_cpu
        The number of CPU to be used.
    """

    if args.nbr_processes:
        nbr_cpu = args.nbr_processes
    else:
        nbr_cpu = multiprocessing.cpu_count()

    if nbr_cpu <= 0:
        parser.error('Number of processes must be > 0.')
    elif nbr_cpu > multiprocessing.cpu_count():
        parser.error('Max number of processes is {}. Got {}.'.format(
            multiprocessing.cpu_count(), nbr_cpu))

    return nbr_cpu


def validate_sh_basis_choice(sh_basis):
    """ Check if the passed sh_basis arg to a fct is right.

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


def assert_inputs_exist(parser, required, optional=None):
    """Assert that all inputs exist. If not, print parser's usage and exit.

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
    args: list
        Argument list.
    required: string or list of paths
        Required paths to be checked.
    optional: string or list of paths
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
                parser.error('Directory {} \n for a given output file '
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
    dirs: list
        Required directory paths to be checked.
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
                    'overwritten. Use -f option if you want to continue.'
                    .format(path))
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


def read_info_from_mb_bdo(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    geometry = root.attrib['type']
    center_tag = root.find('origin')
    flip = [-1, -1, 1]
    center = [flip[0]*float(center_tag.attrib['x'].replace(',', '.')),
              flip[1]*float(center_tag.attrib['y'].replace(',', '.')),
              flip[2]*float(center_tag.attrib['z'].replace(',', '.'))]
    row_list = tree.getiterator('Row')
    radius = [None, None, None]
    for i, row in enumerate(row_list):
        for j in range(0, 3):
            if j == i:
                key = 'col' + str(j+1)
                radius[i] = float(row.attrib[key].replace(',', '.'))
            else:
                key = 'col' + str(j+1)
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

        rot = transfo_dict['AffineTransform_double_3_3'][0:9].reshape((3, 3))
        trans = transfo_dict['AffineTransform_double_3_3'][9:12]
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
