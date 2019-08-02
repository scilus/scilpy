#! /usr/bin/env python
'''
    Script to convert bval/bvec mrtrix style to fsl style.
'''

import argparse
import os

from scilpy.utils.bvec_bval_tools import mrtrix2fsl

def buildArgsParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Convert bval/bvec mrtrix style to fsl style.')

    parser.add_argument('mrtrix_enc', action='store',
                        metavar='mrtrix_enc', type=str,
                        help='Path to gradient directions encoding file.')

    parser.add_argument('fsl_bval', action='store',
                        metavar='fsl_bval', type=str,
                        help='Path to fsl b-value file.')

    parser.add_argument('fsl_bvec', action='store',
                        metavar='fsl_bvec', type=str,
                        help='Path to fsl gradient directions file.')

    parser.add_argument('-f', action='store_true', dest='isForce',
                        help='Force (overwrite output file). [%(default)s]')

    return parser

def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if not os.path.exists(args.mrtrix_enc):
        parser.error('"{0}"'.format(args.mrtrix_enc) +
            " doesn't exist. Please enter an existing file.")

    if os.path.exists(args.fsl_bval):
        if args.isForce:
            print('Overwriting "{0}".'.format(args.fsl_bval))
        else:
            parser.error(
                '"{0}" already exist! Use -f to overwrite it.'
                .format(args.fsl_bval))

    if os.path.exists(args.fsl_bvec):
        if args.isForce:
            print('Overwriting "{0}".'.format(args.fsl_bvec))
        else:
            parser.error('"{0}" already exist! Use -f to overwrite it.'
                .format(args.fsl_bvec))

    mrtrix2fsl(args.mrtrix_enc, args.fsl_bval, args.fsl_bvec)

if __name__ == "__main__":
    main()
