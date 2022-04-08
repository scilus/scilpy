#!/usr/bin/env python3
import argparse
import nibabel as nib

from scilpy.io.utils import add_overwrite_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    # mandatory tracking options
    p.add_argument('in_odf',
                   help='File containing the orientation diffusion function \n'
                        'as spherical harmonics file (.nii.gz). Ex: ODF or '
                        'fODF.')
    p.add_argument('in_mask',
                   help='Tracking mask (.nii.gz).\n'
                        'Tracking will stop outside this mask.')
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    print('hello world')


if __name__ == '__main__':
    main()
