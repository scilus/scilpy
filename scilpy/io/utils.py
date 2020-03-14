# -*- coding: utf-8 -*-

import os
import shutil
import xml.etree.ElementTree as ET

import nibabel as nib
from nibabel.streamlines import TrkFile
import numpy as np
import six

from scilpy.utils.bvec_bval_tools import DEFAULT_B0_THRESHOLD


def link_bundles_and_reference(parser, args, input_tractogram_list):
    """
    Associate the bundle to their reference (if they require a reference)
    Parameters
    ----------
    parser: argparse.ArgumentParser object
        Parser as created by argparse
    args: argparse namespace
        Args as created by argparse
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


def add_json_args(parser):
    g1 = parser.add_argument_group(title='Json options')
    g1.add_argument('--indent',
                    type=int, default=2,
                    help='Indent for json pretty print.')
    g1.add_argument('--sort_keys',
                    action='store_true',
                    help='Sort keys in output json.')


def add_processes_args(parser):
    parser.add_argument('--processes', dest='nbr_processes',
                        metavar='NBR', type=int,
                        help='Number of sub-processes to start. \n'
                        'Default: CPU count')


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
    :param parser: argparse.ArgumentParser object
    :param mandatory: should this argument be mandatory
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
    """
    Assert that all inputs exist. If not, print parser's usage and exit.
    :param parser: argparse.ArgumentParser object
    :param required: string or list of paths
    :param optional: string or list of paths.
                     Each element will be ignored if None
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


def assert_outputs_exist(parser, args, required, optional=None):
    """
    Assert that all outputs don't exist or that if they exist, -f was used.
    If not, print parser's usage and exit.
    :param parser: argparse.ArgumentParser object
    :param args: argparse namespace
    :param required: string or list of paths
    :param optional: string or list of paths.
                     Each element will be ignored if None
    """
    def check(path):
        if os.path.isfile(path) and not args.overwrite:
            parser.error('Output file {} exists. Use -f to force '
                         'overwriting'.format(path))

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for required_file in required:
        check(required_file)
    for optional_file in optional or []:
        if optional_file is not None:
            check(optional_file)


def assert_output_dirs_exist_and_empty(parser, args, *dirs, create_dir=False):
    """
    Assert that all output directories exist.
    If not, print parser's usage and exit.
    If exists and not empty, and -f used, delete dirs.
    :param parser: argparse.ArgumentParser object
    :param args: argparse namespace
    :param dirs: list of paths
    """
    for cur_dir in dirs:
        if not os.path.isdir(cur_dir):
            if not create_dir:
                parser.error('Output directory {} doesn\'t exist.'.format(cur_dir))
            else:
                os.makedirs(cur_dir, exist_ok=True)
        if os.listdir(cur_dir):
            if not args.overwrite:
                parser.error(
                    'Output directory {} isn\'t empty and some files could be '
                    'overwritten. Use -f option if you want to continue.'
                    .format(cur_dir))
            else:
                for the_file in os.listdir(cur_dir):
                    file_path = os.path.join(cur_dir, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)


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
