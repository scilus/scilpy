#! /usr/bin/env python

from __future__ import print_function, division

import argparse
import os.path

import nibabel as nib
import logging


DESCRIPTION = """
    Flip the volume according to the specified axis.
    """


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('input', action='store', metavar='input', type=str,
                   help='Path of the input volume (nifti).')

    p.add_argument('output', action='store', metavar='output', type=str,
                   help='Path of the output volume (nifti).')

    p.add_argument('-x', action='store_true', dest='x',
                   required=False,
                   help='If supplied, flip the x axis.')

    p.add_argument('-y', action='store_true', dest='y',
                   required=False,
                   help='If supplied, flip the y axis.')

    p.add_argument('-z', action='store_true', dest='z',
                   required=False,
                   help='If supplied, flip the z axis.')

    p.add_argument('-f', action='store_true', dest='overwrite', required=False,
                   help='If set, the saved file will be overwritten ' +
                        'if it already exists.')
    return p


def main():

    parser = buildArgsParser()
    args = parser.parse_args()

    if os.path.isfile(args.output):
        if args.overwrite:
            logging.info('Overwriting "{0}".'.format(args.output))
        else:
            parser.error('"{0}" already exists! Use -f to overwrite it.'
                         .format(args.output))

    vol = nib.load(args.input)
    data = vol.get_data()
    affine = vol.get_affine()
    header = vol.get_header()

    if args.x:
        data = data[::-1, ...]

    if args.y:
        data = data[:, ::-1, ...]

    if args.z:
        data = data[:, :, ::-1, ...]

    nib.save(nib.Nifti1Image(data, affine, header), args.output)


if __name__ == "__main__":
    main()
