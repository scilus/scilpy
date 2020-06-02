#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-order gradient according to original table
"""

import argparse
import logging

from dipy.io.gradients import read_bvals_bvecs
import nibabel as nb
import numpy as np

from scilpy.io.utils import (add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.filenames import split_name_with_nii


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('dwi',
                   help='input dwi file')

    p.add_argument('bvec',
                   help='input bvec FSL format')

    p.add_argument('bval',
                   help='input bval FSL format')

    p.add_argument('table',
                   help='original table - first line is skipped')

    p.add_argument('baseName',
                   help='basename output file')

    add_overwrite_arg(p)

    add_verbose_arg(p)

    return p


def valideInputs(oTable, dwis, bvals, bvecs):

    logging.info('Check number of b0s, gradients per shells,\
                  directions overall')

    # Check number of gradients, bvecs, bvals, dwi and oTable
    if len(bvecs) != dwis.shape[3] or len(bvals) != len(oTable):
        raise ValueError('bvec/bval/dwi and original table \
            does not contain the same number of gradients')

    # Check bvals
    tableBVals = np.unique(oTable[:, 3])
    tableDWIShells = tableBVals[tableBVals > 1]
    tableB0Shells = tableBVals[tableBVals < 1]

    dwiShells = np.unique(bvals[bvals > 1])
    b0Shells = np.unique(bvals[bvals < 1])

    if len(tableDWIShells) != len(dwiShells) or\
       len(tableB0Shells) != len(b0Shells):
        raise ValueError('bvec/bval/dwi and original table\
                          does not contain the same shells')

    newIndex = np.zeros(bvals.shape)

    for nBVal in tableBVals:
        currBVal = np.where(bvals == nBVal)[0]
        currBValTable = np.where(oTable[:, 3] == nBVal)[0]

        if len(currBVal) != len(currBValTable):
            raise ValueError('bval/bvec and orginal table does not contain \
                the same number of gradients for shell {0}'.format(nBVal))

        newIndex[currBValTable] = currBVal

    return newIndex.astype(int)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    required_args = [args.dwi, args.bvec, args.bval, args.table]

    baseName, extension = split_name_with_nii(args.dwi)
    output_filenames = [args.baseName + extension,
                        args.baseName + '.bval',
                        args.baseName + '.bvec']

    assert_inputs_exist(parser, required_args)
    assert_outputs_exist(parser, args, output_filenames)

    oTable = np.loadtxt(args.table, skiprows=1)
    bvals, bvecs = read_bvals_bvecs(args.bval, args.bvec)
    dwis = nb.load(args.dwi)

    newIndex = valideInputs(oTable, dwis, bvals, bvecs)

    bvecs = bvecs[newIndex]
    bvals = bvals[newIndex]

    data = dwis.get_data()
    data = data[:, :, :, newIndex]

    nb.save(nb.Nifti1Image(data.astype(dwis.get_data_dtype()), dwis.affine,
                           header=dwis.header), output_filenames[0])
    np.savetxt(args.baseName + '.bval', bvals.reshape(1, len(bvals)), '%d')
    np.savetxt(args.baseName + '.bvec', bvecs.T, '%0.15f')


if __name__ == '__main__':
    main()
