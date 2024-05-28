#!/home/ayat-jinane-g2/src/scilpy/myenv/bin/python3.10

"""
Display a tractogram and its density map (computed from Dipy) in rasmm,
voxmm and vox space with its bounding box.
"""

import argparse

from trx.workflows import tractogram_visualize_overlap


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram', metavar='IN_TRACTOGRAM',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy, trx.')
    p.add_argument('reference',
                   help='Reference anatomy for tck/vtk/fib/dpy file\n'
                        'support (nii or nii.gz).')
    p.add_argument('--remove_invalid', action='store_true',
                   help='Removes invalid streamlines to avoid the density_map'
                        'function to crash.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    tractogram_visualize_overlap(args.in_tractogram, args.reference,
                                 args.remove_invalid)


if __name__ == "__main__":
    main()
