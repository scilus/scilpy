#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import argparse
import logging
import numpy as np

from dipy.io.gradients import read_bvals_bvecs

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             add_b0_thresh_arg,
                             add_tolerance_arg, assert_inputs_exist,
                             assert_gradients_filenames_valid)
from scilpy.gradients.bvec_bval_tools import (check_b0_threshold,
                                              is_normalized_bvecs,
                                              normalize_bvecs,
                                              identify_shells)
from scilpy.gradients.gen_gradient_sampling import (generate_gradient_sampling,
                                                    energy_comparison)
from scilpy.viz.gradients import (plot_proj_shell, build_ms_from_shell_idx)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    p.add_argument(
        'in_gradients', nargs='+',
        help='Path(s) to the gradient file(s). Either FSL '
             '(.bval, .bvec) or MRtrix (.b).')

    p.add_argument(
        '--visualize', action='store_true',
        help='If set, the inputed gradient scheme is displayed, and then the '
             'optimal one.')

    add_b0_thresh_arg(p)
    add_overwrite_arg(p)
    add_tolerance_arg(p)
    add_verbose_arg(p)

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
    scipy_verbose = int(3 - logging.getLogger().getEffectiveLevel()//10)
    opt_bvecs, _ = generate_gradient_sampling(nb_dir_per_shell,
                                              verbose=scipy_verbose)

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

    perc_comp = np.round(opt_energy / energy * 100)
    print('Compared to our reference optimal set of b-vectors, the inputed '
          'b-vectors are {}% less optimally distributed.'.format(perc_comp))
    
    logging.info('The calculated electrostatic-like repulsion energy for the '
                 'optimal b-vectors is: {}'.format(opt_energy))
    logging.info('The calculated electrostatic-like repulsion energy for the '
                 'inputed b-vectors is: {}'.format(energy))

    # TODO: Find a better sentence, and add a warning if the bvecs are too shit


if __name__ == "__main__":
    main()
