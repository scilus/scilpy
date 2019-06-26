#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import six

import nibabel as nib
from nibabel.streamlines import TrkFile

from scilpy.utils.bvec_bval_tools import DEFAULT_B0_THRESHOLD


def add_overwrite_arg(parser):
    parser.add_argument(
        '-f', dest='overwrite', action='store_true',
        help='Force overwriting of the output files.')


def add_force_b0_arg(parser):
    parser.add_argument('--force_b0_threshold', action='store_true',
                        help='If set, the script will continue even if the '
                             'minimum bvalue is suspiciously high ( > {})'
                        .format(DEFAULT_B0_THRESHOLD))


def add_verbose(parser):
    parser.add_argument('-v', action='store_true', dest='isVerbose',
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


def assert_inputs_exist(parser, required, optional=None):
    """
    Assert that all inputs exist. If not, print parser's usage and exit.
    :param parser: argparse.ArgumentParser object
    :param required: list of paths
    :param optional: list of paths. Each element will be ignored if None
    """
    def check(path):
        if not os.path.isfile(path):
            parser.error('Input file {} does not exist'.format(path))

    for required_file in required:
        check(required_file)
    for optional_file in optional or []:
        if optional_file is not None:
            check(optional_file)


def assert_outputs_exists(parser, args, required, optional=None):
    """
    Assert that all outputs don't exist or that if they exist, -f was used.
    If not, print parser's usage and exit.
    :param parser: argparse.ArgumentParser object
    :param args: argparse namespace
    :param required: list of paths
    :param optional: list of paths. Each element will be ignored if None
    """
    def check(path):
        if os.path.isfile(path) and not args.overwrite:
            parser.error('Output file {} exists. Use -f to force '
                         'overwriting'.format(path))

    for required_file in required:
        check(required_file)
    for optional_file in optional or []:
        if optional_file is not None:
            check(optional_file)


def create_header_from_anat(reference, base_filetype=TrkFile):
    """
    Create a valid header for a TRK or TCK file from an reference NIFTI file
    :param reference: Nibabel.nifti or filepath (nii or nii.gz)
    :param base_filetype: Either TrkFile or TckFile from nibabal.streamlines
    """
    if isinstance(reference, six.string_types):
        reference = nib.load(reference)

    new_header = base_filetype.create_empty_header()

    new_header[nib.streamlines.Field.VOXEL_SIZES] = tuple(reference.header.
                                                          get_zooms())[:3]
    new_header[nib.streamlines.Field.DIMENSIONS] = tuple(reference.shape)[:3]
    new_header[nib.streamlines.Field.VOXEL_TO_RASMM] = (reference.header.
                                                        get_best_affine())
    affine = new_header[nib.streamlines.Field.VOXEL_TO_RASMM]

    new_header[nib.streamlines.Field.VOXEL_ORDER] = ''.join(
        nib.aff2axcodes(affine))

    return new_header
