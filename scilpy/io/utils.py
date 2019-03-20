#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
# import six
# import xml.etree.ElementTree as ET
#
# import nibabel as nib
# import numpy as np
#
# from scilpy.utils.bvec_bval_tools import DEFAULT_B0_THRESHOLD


def add_overwrite_arg(parser):
    parser.add_argument(
        '-f', dest='overwrite', action='store_true',
        help='Force overwriting of the output files.')


# def add_force_b0_arg(parser):
#     parser.add_argument('--force_b0_threshold', action='store_true',
#                         help='If set, the script will continue even if the '
#                              'minimum bvalue is suspiciously high ( > {})'
#                         .format(DEFAULT_B0_THRESHOLD))
#
#
# def add_sh_basis_args(parser, mandatory=False):
#     """Add spherical harmonics (SH) bases argument.
#     :param parser: argparse.ArgumentParser object
#     :param mandatory: should this argument be mandatory
#     """
#     choices = ['descoteaux07', 'tournier07']
#     def_val = 'descoteaux07'
#     help_msg = 'Spherical harmonics basis used for the SH coefficients.\nMust ' +\
#                'be either \'descoteaux07\' or \'tournier07\' [%(default)s]:\n' +\
#                '    \'descoteaux07\': SH basis from the Descoteaux et al.\n' +\
#                '                      MRM 2007 paper\n' +\
#                '    \'tournier07\'  : SH basis from the Tournier et al.\n' +\
#                '                      NeuroImage 2007 paper.'
#
#     if mandatory:
#         arg_name = 'sh_basis'
#     else:
#         arg_name = '--sh_basis'
#
#     parser.add_argument(arg_name,
#                         choices=choices, default=def_val,
#                         help=help_msg)
#
#
# def add_tract_producer_arg(parser):
#     parser.add_argument(
#         '--tp', metavar='TRACT_PRODUCER', dest='tracts_producer',
#         choices=['scilpy', 'trackvis'],
#         help='software used to produce the tracts.\nMust be provided when '
#              'processing a .trk file, to be able to guess\nthe corner '
#              'alignment of the file. Can be:\n'
#              '    scilpy: any tracking algorithm from scilpy\n'
#              '    trackvis: any tool in the trackvis family')


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


# def find_roi_sphere_element_by_name(rois_element, sphere_name):
#     child_elements = rois_element.findall('ROI')
#
#     for roi_el in child_elements:
#         if roi_el.attrib['name'] == sphere_name and\
#            roi_el.attrib['type'] == 'Sphere':
#             return roi_el
#
#     return None
#
#
# def read_sphere_info_from_scene(scene_filename, sphere_name):
#     # The trackvis scene format is not valid XML (surprise, surprise...)
#     # because there are 2 "root" elements.
#     # We need to fake it by reading the file ourselves and adding a fake root
#     with open(scene_filename, 'rb') as scene_file:
#         scene_str = '<fake_root>' + scene_file.read() + '</fake_root>'
#
#     # We need to filter out the P## element that Trackvis likes to add to the
#     # scene.
#     coord_start = scene_str.find('<Coordinate>')
#     coord_end = scene_str.find('</Coordinate>')
#     coord_end += len('</Coordinate>')
#     scene_str = scene_str.replace(scene_str[coord_start:coord_end], '')
#
#     root = ET.fromstring(scene_str)
#
#     scene_el = root.find('Scene')
#
#     dim_el = scene_el.find('Dimension')
#
#     x_dim = int(dim_el.attrib.get('x'))
#     y_dim = int(dim_el.attrib.get('y'))
#
#     voxel_order = scene_el.find('VoxelOrder').attrib['current']
#
#     if voxel_order != 'LPS':
#         raise IOError('Scene file voxel order was not LPS, unsupported.')
#
#     rois_el = scene_el.find('ROIs')
#
#     sphere_el = find_roi_sphere_element_by_name(rois_el, sphere_name)
#
#     if sphere_el is None:
#         raise IOError(
#             'Scene file did not contain a sphere called {0}'.format(sphere_name))
#
#     center_el = sphere_el.find('Center')
#     center_x = float(center_el.attrib['x'])
#     center_y = float(center_el.attrib['y'])
#     center_z = float(center_el.attrib['z'])
#
#     size_el = sphere_el.find('Radius')
#     sphere_size = float(size_el.attrib['value'])
#
#     # Need to flip the X and Y coordinates since they were in LPS
#     center_x = x_dim - 1 - center_x
#     center_y = y_dim - 1 - center_y
#
#     return np.array([center_x, center_y, center_z]), sphere_size
#
#
# def read_info_from_mb_bdo(filename):
#     tree = ET.parse(filename)
#     root = tree.getroot()
#     geometry = root.attrib['type']
#     center_tag = root.find('origin')
#     flip = [-1, -1, 1]
#     center = [flip[0]*float(center_tag.attrib['x'].replace(',', '.')),
#               flip[1]*float(center_tag.attrib['y'].replace(',', '.')),
#               flip[2]*float(center_tag.attrib['z'].replace(',', '.'))]
#     row_list = tree.getiterator('Row')
#     radius = [None, None, None]
#     for i, row in enumerate(row_list):
#         for j in range(0, 3):
#             if j == i:
#                 key = 'col' + str(j+1)
#                 radius[i] = float(row.attrib[key].replace(',', '.'))
#             else:
#                 key = 'col' + str(j+1)
#                 value = float(row.attrib[key].replace(',', '.'))
#                 if abs(value) > 0.01:
#                     raise ValueError('Does not support rotation, for now only \n'
#                                      'SO aligned on the X,Y,Z axis are supported')
#     radius = np.asarray(radius, dtype=np.float32)
#     center = np.asarray(center, dtype=np.float32)
#     return geometry, radius, center
#
#
# def assert_outputs_dir_exists_and_empty(parser, args, *dirs):
#     """
#     Assert that all output folder exist If not, print parser's usage and exit.
#     :param parser: argparse.ArgumentParser object
#     :param args: argparse namespace
#     :param dirs: list of paths
#     """
#     for path in dirs:
#         if not os.path.isdir(path):
#             parser.error('Output directory {} doesn\'t exist.'.format(path))
#         if os.listdir(path) and not args.overwrite:
#             parser.error(
#                 'Output directory {} isn\'t empty and some files could be '
#                 'overwritten. Use -f option if you want to continue.'
#                 .format(path))
#
#
# def create_header_from_anat(reference):
#     if isinstance(reference, six.string_types):
#         reference = nib.load(reference)
#     new_header = nib.streamlines.TrkFile.create_empty_header()
#
#     new_header[nib.streamlines.Field.VOXEL_SIZES] = tuple(reference.header.
#                                                           get_zooms())[:3]
#     new_header[nib.streamlines.Field.DIMENSIONS] = tuple(reference.shape)[:3]
#     new_header[nib.streamlines.Field.VOXEL_TO_RASMM] = (reference.header.
#                                                         get_best_affine())
#     affine = new_header[nib.streamlines.Field.VOXEL_TO_RASMM]
#
#     new_header[nib.streamlines.Field.VOXEL_ORDER] = ''.join(
#         nib.aff2axcodes(affine))
#
#     return new_header
#
#
# def verify_header_compatibility(obj_1, obj_2, enforce_dimensions=True):
#     """
#     Verify that headers from nifti or trkfile are consistent with each other
#     :param obj_1: instance of either a Nifti1Image or a TrkFile
#     :param obj_2: instance of either a Nifti1Image or a TrkFile
#     :param enforce_dimensions: Verify the 'shape' in addition to the affine
#     """
#     if isinstance(obj_1, nib.nifti1.Nifti1Image):
#         affine_1 = obj_1.affine
#         dimension_1 = obj_1.shape[0:3]
#     elif isinstance(obj_1, nib.streamlines.trk.TrkFile):
#         affine_1 = obj_1.header["voxel_to_rasmm"]
#         dimension_1 = obj_1.header["dimensions"]
#
#     if isinstance(obj_2, nib.nifti1.Nifti1Image):
#         affine_2 = obj_2.affine
#         dimension_2 = obj_2.shape[0:3]
#     elif isinstance(obj_2, nib.streamlines.trk.TrkFile):
#         affine_2 = obj_2.header["voxel_to_rasmm"]
#         dimension_2 = obj_2.header["dimensions"]
#
#     # Minimal verification to make sure both datasets were generated together
#     if not np.allclose(affine_1, affine_2):
#         return False
#
#     if enforce_dimensions and not np.array_equal(dimension_1, dimension_2):
#         return False
#
#     return True
