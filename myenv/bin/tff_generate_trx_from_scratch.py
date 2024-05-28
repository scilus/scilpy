#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10
# -*- coding: utf-8 -*-

"""
Generate TRX file from a collection of CSV, TXT or NPY files by individually
specifiying positions, offsets, data_per_vertex, data_per_streamlines,
groups and data_per_group. Each file must have its data type specified by the
users.

A reference file must be provided (NIFTI) and the option --verify_invalid will
remove invalid streamlines (outside of the bounding box in VOX space).

All dimensions (nbr_vertices and nbr_streamlines) and groups/dpg must match
otherwise the script will (likely) crash.

Each instance of --dps, --dpv, --groups require 2 arguments (FILE, DTYPE).
--dpg requires 3 arguments (GROUP, FILE, DTYPE).
The choice of DTYPE are:
    - (u)int8, (u)int16, (u)int32, (u)int64
    - float16, float32, float64
    - bool

Example command:
tff_generate_trx_from_scratch.py fa.nii.gz generated.trx -f \
    --positions test_npy/positions.npy --positions_dtype float16 \
    --offsets test_npy/offsets.npy --offsets_dtype uint32 \
    --dpv test_npy/dpv_cx.npy uint8 \
    --dpv test_npy/dpv_cy.npy uint8 \
    --dpv test_npy/dpv_cz.npy uint8 \
    --dps test_npy/dps_algo.npy uint8 \
    --dps test_npy/dps_cw.npy float64 \
    --groups test_npy/g_AF_L.npy int32 \
    --groups test_npy/g_AF_R.npy int32 \
    --dpg g_AF_L test_npy/dpg_AF_L_mean_fa.npy float32 \
    --dpg g_AF_R test_npy/dpg_AF_R_mean_fa.npy float32 \
    --dpg g_AF_L test_npy/dpg_AF_L_volume.npy float32
"""

import argparse
import os

from trx.workflows import generate_trx_from_scratch


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('reference',
                   help='Reference anatomy for tck/vtk/fib/dpy file\n'
                   'support (.nii or .nii.gz).')
    p.add_argument('out_tractogram', metavar='OUT_TRACTOGRAM',
                   help='Output filename. Format must be one of\n'
                        'trk, tck, vtk, fib, dpy, trx.')

    p1 = p.add_argument_group(title='Positions options')
    p1.add_argument('--positions', metavar='POSITIONS',
                    help='Binary file containing the streamlines coordinates.'
                         '\nMust be Nx3 (.npy)')
    p1.add_argument('--offsets', metavar='OFFSETS',
                    help='Binary file containing the streamlines offsets (.npy)')
    p1.add_argument('--positions_csv', metavar='POSITIONS',
                    help='CSV file containing the streamlines coordinates.'
                         '\nRows for each streamlines organized as x1,y1,z1,\n'
                         'x2,y2,z2,...,xN,yN,zN')
    p1.add_argument('--space', choices=['RASMM', 'VOXMM', 'VOX'],
                    default='RASMM',
                    help='Space in which the coordinates are declared.'
                         '[%(default)s]\nNon-default option requires Dipy.')
    p1.add_argument('--origin', choices=['NIFTI', 'TRACKVIS'],
                    default='NIFTI',
                    help='Origin in which the coordinates are declared. '
                         '[%(default)s]\nNon-default option requires Dipy.')
    p2 = p.add_argument_group(title='Data type options')
    p2.add_argument('--positions_dtype', default='float32',
                    choices=['float16', 'float32', 'float64'],
                    help='Specify the datatype for positions for trx. '
                         '[%(default)s]')
    p2.add_argument('--offsets_dtype', default='uint64',
                    choices=['uint32', 'uint64'],
                    help='Specify the datatype for offsets for trx. '
                         '[%(default)s]')

    p3 = p.add_argument_group(title='Streamlines metadata options')
    p3.add_argument('--dpv', metavar=('FILE', 'DTYPE'), nargs=2,
                    action='append',
                    help='Binary file containing data_per_vertex.\n Must have'
                         'NB_VERTICES as first dimension (.npy)')
    p3.add_argument('--dps', metavar=('FILE', 'DTYPE'), nargs=2,
                    action='append',
                    help='Binary file containing data_per_vertex.\n Must have'
                         'NB_STREAMLINES as first dimension (.npy)')
    p3.add_argument('--groups', metavar=('FILE', 'DTYPE'), nargs=2,
                    action='append',
                    help='Binary file containing a sparse group (indices).\n '
                         'Indices should be lower than NB_STREAMLINES (.npy)')
    p3.add_argument('--dpg', metavar=('GROUP', 'FILE', 'DTYPE'), nargs=3,
                    action='append',
                    help='Binary file containing data_per_group.\n Must have'
                         '(1,) as first dimension (.npy)')

    p.add_argument('--verify_invalid', action='store_true',
                   help='Verify that the positions are all valid.\n'
                        'None outside of the bounding box in VOX space.\n'
                        'Requires Dipy (due to use of SFT).')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_tractogram) and not args.overwrite:
        raise IOError('{} already exists, use -f to overwrite.'.format(
            args.out_tractogram))

    if not args.positions and not args.positions_csv:
        parser.error('At least one positions options must be used.')
    if args.positions_csv and args.positions:
        parser.error('Cannot use both positions options.')
    if args.positions and args.offsets is None:
        parser.error('--offsets must be provided if --positions is used.')
    if args.offsets and args.positions is None:
        parser.error('--positions must be provided if --offsets is used.')

    generate_trx_from_scratch(args.reference, args.out_tractogram,
                              positions_csv=args.positions_csv,
                              positions=args.positions, offsets=args.offsets,
                              positions_dtype=args.positions_dtype,
                              offsets_dtype=args.offsets_dtype,
                              space_str=args.space, origin_str=args.origin,
                              verify_invalid=args.verify_invalid,
                              dpv=args.dpv, dps=args.dps,
                              groups=args.groups, dpg=args.dpg)


if __name__ == "__main__":
    main()
