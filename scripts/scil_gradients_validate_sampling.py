#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate the sampling of a gradient table, in terms of how well distributed on
the sphere the b-vectors are.

To do so, the script compares the electrostatic-like repulsion energy [1] of
the inputed b-vectors with the energy of optimally distributed b-vectors. The
same number of directions per shell (the script supports multi-shell) as the
input b-vectors are used to generate the optimal b-vectors as in [1]. It is
possible that the inputed b-vectors are better distributed than the optimal
ones.

The script starts by looking for b0s, to remove them from the analysis. Then,
it looks for duplicate b-vectors. Finally, both energies are computed and
compared as the ratio between the inputed b-vectors' energy and the optimal
b-vectors' energy (input_energy/optimal_energy). Above a given maximum ratio
value, the script raises a warning.

The user might want to use the -v verbose option to see the computed energies.
The --visualize option displays both the inputed and optimal b-vectors on a
single shell.
"""

import argparse
import logging
import numpy as np

from dipy.io.gradients import read_bvals_bvecs

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             add_b0_thresh_arg,
                             add_tolerance_arg, assert_inputs_exist,
                             assert_gradients_filenames_valid)
from scilpy.gradients.bvec_bval_tools import (is_normalized_bvecs,
                                              normalize_bvecs,
                                              identify_shells)
from scilpy.gradients.gen_gradient_sampling import (generate_gradient_sampling,
                                                    energy_comparison)
from scilpy.viz.gradients import (plot_proj_shell, build_ms_from_shell_idx)


EPILOG = """
References: [1] Emmanuel Caruyer, Christophe Lenglet, Guillermo Sapiro,
Rachid Deriche. Design of multishell gradient sampling with uniform coverage
in diffusion MRI. Magnetic Resonance in Medicine, Wiley, 2013, 69 (6),
pp. 1534-1540. <http://dx.doi.org/10.1002/mrm.24736>
    """


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
        epilog=EPILOG)
    p.add_argument('in_gradients', nargs='+',
                   help='Path(s) to the gradient file(s). Either FSL '
                        '(.bval, .bvec) or MRtrix (.b).')

    p.add_argument(
        '--max_ratio', default=1.25, type=float,
        help='Maximum value for the ratio between the inputed b-vectors\' '
             'energy \nand the optimal b-vectors\' energy '
             '(input_energy/optimal_energy). [%(default)s]')

    p.add_argument(
        '--visualize', action='store_true',
        help='If set, the inputed gradient scheme is displayed, and then the '
             'optimal one.')

    add_b0_thresh_arg(p)
    add_tolerance_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Perform checks on input files and load bvals and bvecs
    assert_inputs_exist(parser, args.in_gradients)
    if len(args.in_gradients) == 2:
        assert_gradients_filenames_valid(parser, args.in_gradients, True)
        in_gradients = args.in_gradients
        in_gradients.sort()
        bvals, bvecs = read_bvals_bvecs(in_gradients[0],
                                        in_gradients[1])
    elif len(args.in_gradients) == 1:
        assert_gradients_filenames_valid(parser, args.in_gradients, False)
        bvecs = np.genfromtxt(args.in_gradients, delimiter=' ')[:, :3]
        bvals = np.genfromtxt(args.in_gradients, delimiter=' ')[:, 3]
    else:
        parser.error('Depending on the gradient format, you should have '
                     'two files for FSL format and one file for MRtrix')

    # Check and remove b0s
    # Note: this part will become check_b0_threshold once it is fixed
    if args.b0_threshold > 20:
        logging.warning(
            'Your defined b0 threshold is {}. This is suspicious. We '
            'recommend using volumes with bvalues no higher than {} as b0s'
            .format(args.b0_threshold, 20))

    if bvals.min() < 0:
        logging.warning(
            'Warning: Your dataset contains negative b-values (minimal bvalue '
            'of {}). This is suspicious. We recommend you check your data.'
            .format(bvals.min()))

    if bvals.min() > args.b0_threshold:
            logging.warning(
                'Your minimal bvalue ({}) is above the threshold ({}). Please '
                'check your data to ensure everything is correct.\n'
                'You may also increase the threshold with --b0_threshold. '
                'The script will continue without b0s.'
                .format(bvals.min(), args.b0_threshold))
    bvecs = bvecs[bvals > args.b0_threshold]
    bvals = bvals[bvals > args.b0_threshold]

    # Checking bvecs are normalized
    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)

    # Find shells and duplicate b-vectors
    ubvals, shell_idx = identify_shells(bvals, tol=args.tolerance, sort=True)
    nb_shells = len(ubvals)
    nb_dir_per_shell = [np.sum(shell_idx == idx) for idx in range(nb_shells)]

    ubvecs = np.unique(bvecs, axis=0)
    nb_ubvecs = len(ubvecs)
    if len(bvecs) != nb_ubvecs:
        raise ValueError('{} b-vectors have the same direction as others, '
                         'which is suboptimal. There is most likely a problem '
                         'with the gradient table.'
                         .format(len(bvecs) - nb_ubvecs))

    # Compute optimally distributed directions
    opt_bvecs, _ = generate_gradient_sampling(nb_dir_per_shell,
                                              verbose=0)

    # Visualize the gradient schemes
    if args.visualize:
        viz_bvecs = build_ms_from_shell_idx(bvecs, shell_idx)
        viz_opt_bvecs = build_ms_from_shell_idx(opt_bvecs, shell_idx)
        plot_proj_shell(viz_bvecs, use_sym=True, title="Inputed b-vectors")
        plot_proj_shell(viz_opt_bvecs, use_sym=True,
                        title="Optimized b-vectors")

    # Compute the energy for both the input bvecs and optimal bvecs.
    energy, opt_energy = energy_comparison(bvecs, opt_bvecs, nb_shells,
                                           nb_dir_per_shell)
    
    logging.info('\nThe quality of inputed b-vectors is assessed by computing '
                 'their electrostatic-like repulsion \nenergy and comparing '
                 'it with the energy of a reference optimal set of b-vectors.')
    logging.info('\nEnergy for the optimal b-vectors: {}'
                 '\nEnergy for the inputed b-vectors: {}'
                 .format(np.round(opt_energy, decimals=3),
                         np.round(energy, decimals=3)))
    e_ratio = energy / opt_energy
    if e_ratio > args.max_ratio:
        logging.warning('\nThe inputed b-vectors seem to be ill-distributed '
                        'on the sphere. \nTheir energy is {} times higher '
                        'than the optimal energy.'
                        .format(np.round(e_ratio, decimals=3)))


if __name__ == "__main__":
    main()
