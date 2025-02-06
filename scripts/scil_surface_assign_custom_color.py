"""

"""

import argparse
import logging

from dipy.io.surface import load_surface, save_surface
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from scilpy.io.utils import (assert_inputs_exist,
                             assert_outputs_exist,
                             add_overwrite_arg,
                             add_verbose_arg,
                             add_reference_arg,
                             add_surface_spatial_arg,
                             add_vtk_legacy_arg,
                             convert_stateful_str_to_enum,
                             load_matrix_in_any_format)
from scilpy.viz.color import get_lookup_table


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_surface',
                   help='Input surface(s) (VTK + PIAL + GII supported).')
    p.add_argument('out_surface',
                   help='Output surface(s) (VTK supported).')

    g1 = p.add_argument_group(title='Coloring method')
    p1 = g1.add_mutually_exclusive_group(required=True)
    p1.add_argument('--load_dpp', metavar='DPP_FILE',
                    help='Load data per point (scalar) for coloring')
    p1.add_argument('--from_anatomy', metavar='FILE',
                    help='Use the voxel data for coloring,\n'
                         'linear scaling from minmax.')

    g2 = p.add_argument_group(title='Coloring options')
    g2.add_argument('--colormap', default='jet',
                    help='Select the colormap for colored trk (dps/dpp) '
                    '[%(default)s].\nUse two Matplotlib named color separeted '
                    'by a - to create your own colormap.')
    g2.add_argument('--log', action='store_true',
                    help='Apply a base 10 logarithm for colored trk (dps/dpp).')

    add_surface_spatial_arg(p)
    add_vtk_legacy_arg(p)
    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Verifications
    assert_inputs_exist(parser, args.in_surface, args.reference)
    assert_outputs_exist(parser, args, args.out_surface)
    convert_stateful_str_to_enum(args)

    # Loading
    sfs = load_surface(args.in_surface, args.reference,
                       from_space=args.source_space,
                       from_origin=args.source_origin)

    cmap = get_lookup_table(args.colormap)

    expected_shape = len(sfs.vertices)
    if args.load_dpp:
        data = np.squeeze(load_matrix_in_any_format(args.load_dpp))
        if len(data) != expected_shape:
            parser.error('Wrong dpp size! Expected a total of {} points, '
                         'but got {}'.format(expected_shape, len(data)))
    else:  # args.from_anatomy:
        data = nib.load(args.from_anatomy).get_fdata()
        sfs.to_vox()
        data = map_coordinates(data, sfs.vertices.T, order=0)

    # Simple post-processing
    if args.log:
        data = np.log10(data + 1e-3)
    data -= data.min()
    data /= data.max()
    data = cmap(data) * 255
    data = data.astype(np.uint8)

    sfs.data_per_point['RGB'] = data
    save_surface(sfs, args.out_surface,
                 to_space=args.destination_space,
                 to_origin=args.destination_origin,
                 legacy_vtk_format=args.legacy_vtk_format)


if __name__ == '__main__':
    main()
