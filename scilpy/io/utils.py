#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import six
import xml.etree.ElementTree as ET

from dipy.io.streamline import load_tractogram
import nibabel as nib
from nibabel.streamlines import TrkFile
import numpy as np

from scilpy.utils.bvec_bval_tools import DEFAULT_B0_THRESHOLD


def add_reference(parser):
    parser.add_argument('--reference',
                        help='Reference anatomy for tck/vtk/fib/dpy file\n'
                        'support (.nii or .nii.gz).')


def load_tractogram_with_reference(parser, args, filepath,
                                   bbox_check=True):
    _, ext = os.path.splitext(filepath)
    if ext == '.trk':
        sft = load_tractogram(filepath, 'same')
    elif ext in ['.tck', '.fib', '.vtk', '.dpy']:
        if args.reference is None:
            parser.error('--reference is required for this file format '
                         '{}.'.format(filepath))
        sft = load_tractogram(filepath, args.reference,
                              bbox_valid_check=bbox_check)
    else:
        parser.error('{} is an unsupported file format'.format(filepath))

    return sft


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


def assert_outputs_exist(parser, args, required, optional=None):
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
