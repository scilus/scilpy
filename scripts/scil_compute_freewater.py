#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute Free Water maps [1] using AMICO.
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
    [1] Pasternak 0, Sochen N, Gur Y, Intrator N, Assaf Y.
        Free water elimination and mapping from diffusion mri.
        Magn Reson Med. 62 (3) (2009) 717-730.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG)

    p.add_argument('dwi',
                   help='DWI file')

    p.add_argument('--mask',
                   help='Mask filename')

    g = p.add_argument_group('Gradients / scheme')
    g.add_argument('--bval',
                   help='Bval filename, in FSL format')
    g.add_argument('--bvec',
                   help='Bvec filename, in FSL format')
    g.add_argument('--scheme_file',
                   help='If a scheme file already exists, '
                        'can replace --bval/--bvec.')
    g.add_argument('--bstep', type=int, nargs='+',
                   help='List of bvals in your data')

    p.add_argument('--para_diff', type=float, default=1.5e-3,
                   help='Axial diffusivity (AD) in the CC. [%(default)s]')
    p.add_argument('--iso_diff', type=float, default=3e-3,
                   help='Mean diffusivity (MD) in ventricles. [%(default)s]')
    p.add_argument('--perp_diff_min', type=float, default=0.1e-3,
                   help='Radial diffusivity (RD) minimum. [%(default)s]')
    p.add_argument('--perp_diff_max', type=float, default=0.7e-3,
                   help='Radial diffusivity (RD) maximum. [%(default)s]')

    p.add_argument('--lambda1', default=0.0, type=float,
                   help='First regularization parameter. [%(default)s]')
    p.add_argument('--lambda2', default=1e-3, type=float,
                   help='Second regularization parameter. [%(default)s]')

    p.add_argument('--mouse', action='store_true',
                   help='If set, use mouse fitting profile.')

    p.add_argument('--output_dir',
                   help='Output directory for the Free Water results. '
                        '[current_directory]')

    p.add_argument('--processes', type=int, default=1,
                   help='Number of processes used to compute Free Water. '
                        'Right now, better performance with 1. [%(default)s]')

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    required_in = [args.dwi]

    if not any([args.bval, args.bvec, args.scheme_file]):
        parser.error('Need to provide either [--bval, --bvec] or '
                     '--scheme_file.')

    if (args.bval or args.bvec) and args.scheme_file:
        parser.error('Can only provide [--bval, --bvec] or --scheme_file.')

    use_scheme_file = False
    if not args.scheme_file:
        if (args.bval and not args.bvec) or (args.bvec and not args.bval):
            parser.error('Need to specify both bvec and bval.')
        required_in.extend([args.bval, args.bvec])
    else:
        required_in.append(args.scheme_file)
        use_scheme_file = True

    assert_inputs_exist(parser, required_in, args.mask)

    out_dir = ''
    if args.output_dir:
        if not os.path.isdir(args.output_dir):
            parser.error("Output directory doesn't exist.")
        out_dir = args.output_dir

    basic_out_files = ['dwi_fw_corrected.nii.gz', 'FIT_dir.nii.gz',
                       'FIT_FiberVolume.nii.gz', 'FIT_FW.nii.gz',
                       'FIT_nrmse.nii.gz']
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
    ae.set_model("FreeWater")

    model_type = 'Human'
    if args.mouse:
        model_type = 'Mouse'

    ae.model.set(args.para_diff,
                 np.linspace(args.perp_diff_min,
                             args.perp_diff_max,
                             10),
                 [args.iso_diff],
                 model_type)

    ae.set_solver(lambda1=args.lambda1, lambda2=args.lambda2)

    ae.generate_kernels(regenerate=True)

    # Load the precomputed kernels at high resolution
    ae.load_kernels()

    # Set number of processes
    solver_params = ae.get_config('solver_params')
    solver_params['numThreads'] = args.processes
    ae.set_config('solver_params', solver_params)

    ae.set_config('doNormalizeSignal', True)
    ae.set_config('doKeepb0Intact', False)
    ae.set_config('doComputeNRMSE', True)
    ae.set_config('doSaveCorrectedDWI', True)

    # Model fit
    ae.fit()

    # Save the results
    ae.save_results()

    # Copy the output files
    amico_def_out = [os.path.join('AMICO/FreeWater/', f)
                     for f in basic_out_files]

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
