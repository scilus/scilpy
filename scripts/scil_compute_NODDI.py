#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute NODDI [1] maps using AMICO.
"""

import argparse
import os
import shutil
import tempfile

try:
    import amico
except ImportError as e:
    e.args += ("AMICO not installed and configured. "
               "Could use a precompiled container.",)
    raise e
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)

amico.core.setup()

EPILOG = """
Reference:
    [1] Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC.
        NODDI: practical in vivo neurite orientation dispersion
        and density imaging of the human brain.
        NeuroImage. 2012 Jul 16;61:1000-16.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('dwi',
                   help='DWI file acquired with a NODDI compatible protocol')

    p.add_argument('--mask',
                   help='Mask filename')

    g = p.add_argument_group('Gradients / scheme')
    g.add_argument('--bval',
                   help='Bval filename, in FSL format')
    g.add_argument('--bvec',
                   help='Bvec filename, in FSL format')
    g.add_argument('--scheme_file',
                   help='AMICO scheme file '
                        'can replace --bval/--bvec.')
    g.add_argument('--bstep', type=int, nargs='+',
                   help='List of unique bvals in your data. It prevents errors'
                        ' when each bval is around the actual bval')

    p.add_argument('--para_diff', type=float, default=1.7e-3,
                   help='Axial diffusivity (AD) in the CC. [%(default)s]')
    p.add_argument('--iso_diff', type=float, default=3e-3,
                   help='Mean diffusivity (MD) in ventricles. [%(default)s]')
    p.add_argument('--lambda1', type=float, default=2,
                   help='First regularization parameter. [%(default)s]')
    p.add_argument('--lambda2', type=float, default=1e-3,
                   help='Second regularization parameter. [%(default)s]')

    p.add_argument('--output_dir',
                   help='Output directory for the NODDI results. '
                        '[%(default)s]')

    p.add_argument('--processes', type=int, default=1,
                   help='Number of processes used to compute NODDI. Right now,'
                        'better performance with 1. [%(default)s]')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    required_in = [args.dwi]

    use_scheme_file = False

    if not any([args.bval, args.bvec, args.scheme_file]):
        parser.error('Need to provide either [--bval, --bvec] or '
                     '--scheme_file.')

    if not args.scheme_file:
        if not all([args.bval, args.bvec, args.bstep]):
            parser.error('Need to specify both bvec, bval and bstep.')
        required_in.extend([args.bval, args.bvec])
    elif any([args.bval, args.bvec, args.bstep]):
        parser.error('Can only provide [--bval, --bvec, --bstep] or '
                     '--scheme_file.')
    else:
        required_in.append(args.scheme_file)
        use_scheme_file = True

    assert_inputs_exist(parser,
                        required_in,
                        args.mask)

    out_dir = ''
    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            parser.error("Output directory doesn't exist.")
        out_dir = args.output_dir

    basic_out_files = ['FIT_dir.nii.gz', 'FIT_ICVF.nii.gz',
                       'FIT_ISOVF.nii.gz', 'FIT_OD.nii.gz']
    out_files = [os.path.join(out_dir, f) for f in basic_out_files]

    assert_outputs_exist(parser, args, out_files)

    if args.processes <= 0:
        parser.error('Number of processes cannot be <= 0.')
    elif args.processes > 1:
        import multiprocessing
        if args.processes > multiprocessing.cpu_count():
            parser.error('Max number of processes is {}. Got {}.'.format(
                multiprocessing.cpu_count(), args.processes))

    # Load the data
    ae = amico.Evaluation('./', './')

    # Generage a scheme file from the bvals and bvecs files
    if use_scheme_file:
        scheme_filename = args.scheme_file
    else:
        scheme_filename = tempfile.mkstemp(suffix='.scheme',
                                           text=True,
                                           dir='././')
        scheme_filename = scheme_filename[1]
        bstep = args.bstep
        amico.util.fsl2scheme(args.bval, args.bvec, scheme_filename, bstep)

    # Load the data
    ae.load_data(dwi_filename=args.dwi,
                 scheme_filename=scheme_filename,
                 mask_filename=args.mask)

    # Compute the response functions
    ae.set_model("NODDI")

    intra_vol_frac = np.linspace(0.1, 0.99, 12)
    intra_orient_distr = np.hstack((np.array([0.03, 0.06]),
                                    np.linspace(0.09, 0.99, 10)))

    ae.model.set(args.para_diff, args.iso_diff,
                 intra_vol_frac, intra_orient_distr, False)
    ae.set_solver(lambda1=args.lambda1, lambda2=args.lambda2)

    ae.generate_kernels(regenerate=True)

    # Load the precomputed kernels at high resolution
    ae.load_kernels()

    # Set number of processes
    solver_params = ae.get_config('solver_params')
    solver_params['numThreads'] = args.processes
    ae.set_config('solver_params', solver_params)

    # Model fit
    ae.fit()

    # Save the results
    ae.save_results()

    # Copy the output files
    amico_def_out = [os.path.join('AMICO/NODDI/', f) for f in basic_out_files]

    for in_f, out_f in zip(amico_def_out, out_files):
        shutil.move(in_f, out_f)

    if args.output_dir != "AMICO":
        shutil.rmtree("./AMICO")

    # Moving back to the original directory
    if not use_scheme_file:
        os.unlink(scheme_filename)

    shutil.rmtree("./kernels")


if __name__ == "__main__":
    main()
