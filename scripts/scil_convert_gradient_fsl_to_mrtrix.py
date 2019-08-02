#! /usr/bin/env python
'''
    Script to convert bval/bvec fsl style to mrtrix style.
'''

import argparse
import os

from scilpy.utils.bvec_bval_tools import fsl2mrtrix

def buildArgsParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Convert bval/bvec fsl style to mrtrix style.')

    parser.add_argument('fsl_bval', action='store', metavar='fsl_bval', type=str,
                        help='path to fsl b-value file.')

    parser.add_argument('fsl_bvec', action='store', metavar='fsl_bvec', type=str,
                        help='path to fsl gradient directions file.')

    parser.add_argument('mrtrix_enc', action='store', metavar='mrtrix_enc', type=str,
                        help='path to gradient directions encoding file.')

    parser.add_argument('-f', action='store_true', dest='isForce',
                        help='Force (overwrite output file). [%(default)s]')

    return parser

def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if not os.path.exists(args.fsl_bval):
        parser.error('"{0}"'.format(args.fsl_bval) +
            " doesn't exist. Please enter an existing file.")

    if not os.path.exists(args.fsl_bvec):
        parser.error('"{0}"'.format(args.fsl_bvec) +
            " doesn't exist. Please enter an existing file.")

    if os.path.exists(args.mrtrix_enc):
        if args.isForce:
            print('Overwriting "{0}".'.format(args.mrtrix_enc))
        else:
            parser.error('"{0}" already exist! Use -f to overwrite it.'
                .format(args.mrtrix_enc))

    fsl2mrtrix(args.fsl_bval, args.fsl_bvec, args.mrtrix_enc)

if __name__ == "__main__":
    main()