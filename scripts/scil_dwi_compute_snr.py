#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute signal to noise ratio (SNR) in a region of interest (ROI)
of a DWI volume.

It will compute the SNR for all DWI volumes of the input image seperately.
The output will contain the SNR which is the ratio of
mean(signal) / std(noise).
The mean of the signal is computed inside the mask.
The standard deviation of the noise is estimated inside the noise_mask
or inside the same mask if a noise_map is provided.
If it's not supplied, it will be estimated using the data outside the brain,
computed with Dipy medotsu

If verbose is True, the SNR for every DWI volume will be output.

This works best in a well-defined ROI such as the corpus callosum.
It is heavily dependent on the ROI and its quality.

We highly recommend using a noise_map if you can acquire one.
See refs [1, 2] that describe the noise map acquisition.
[1] St-Jean, et al (2016). Non Local Spatial and Angular Matching...
    https://doi.org/10.1016/j.media.2016.02.010
[2] Reymbaut, et al (2021). Magic DIAMOND...
    https://doi.org/10.1016/j.media.2021.101988

Formerly: scil_snr_in_roi.py
"""

import argparse
import logging
import os

from dipy.io.gradients import read_bvals_bvecs
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from scilpy.io.utils import (add_json_args, add_overwrite_arg,
                             add_verbose_arg,
                             assert_inputs_exist)
from scilpy.utils.filenames import split_name_with_nii
from scilpy.image.volume_operations import compute_snr


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_dwi',
                   help='Path of the input diffusion volume.')

    p.add_argument('in_bval',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('in_bvec',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('in_mask',
                   help='Binary mask of the region used to estimate SNR.')

    g1 = p.add_argument_group(title='Masks options')
    mask = g1.add_mutually_exclusive_group()
    mask.add_argument('--noise_mask',
                      help='Binary mask used to estimate the noise '
                           'from the DWI.')
    mask.add_argument('--noise_map',
                      help='Noise map.')

    p.add_argument('--b0_thr',
                   type=float, default=0.0,
                   help='All b-values with values less than or equal '
                        'to b0_thr are considered as b0s i.e. without '
                        'diffusion weighting. [%(default)s]')
    p.add_argument('--out_basename',
                   help='Path and prefix for the various saved file.')
    p.add_argument('--split_shells',
                   action='store_true',
                   help='SNR will be split into shells.')

    add_json_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose == "WARNING":
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval,
                                 args.in_bvec, args.in_mask],
                        [args.noise_mask, args.noise_map])

    basename, ext = split_name_with_nii(os.path.basename(args.in_dwi))

    if args.out_basename:
        basename = args.out_basename

    logging.info('Basename: {}'.format(basename))

    # Loadings inputs.
    dwi = nib.load(args.in_dwi)
    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
    mask = nib.load(args.in_mask)

    if args.noise_mask:
        noise_mask = nib.load(args.noise_mask)
        noise_map = None
    else:
        noise_map = nib.load(args.noise_map)
        noise_mask = None

    automatic_mask_discovery = noise_mask is None and noise_map is None
    values, noise_mask = compute_snr(dwi, bvals, bvecs, args.b0_thr,
                                     mask, noise_mask=noise_mask,
                                     noise_map=noise_map,
                                     split_shells=args.split_shells)

    if automatic_mask_discovery:
        filename = basename + '_noise_mask.nii.gz'
        logging.info("Saving computed noise mask as {}".format(filename))
        nib.save(nib.Nifti1Image(noise_mask, dwi.affine), filename)

    df = pd.DataFrame.from_dict(values).T

    if args.split_shells:
        for curr_shell in np.unique(df['bval']):
            curr_values = df.loc[df['bval'] == curr_shell]['snr']
            plt.plot(curr_values,
                     marker='+', linestyle='--')
            plt.legend(["SNR for bval: " + str(curr_shell)])
            plt.xlabel("Directions")
            plt.xlim([-1, len(df)])
            plt.ylim([0, np.max(df['snr'])])
            plt.ylabel("Estimated SNR")
            plt.text(1, -9, 'Min SNR = ' + str(np.min(curr_values)))
            plt.text(1, -13, 'Max SNR = ' + str(np.max(curr_values)))
            plt.text(1, -17, 'Mean SNR = ' + str(np.mean(curr_values)))
            out_png = basename + "_graph_SNR_bval_" + str(curr_shell) + ".png"
            plt.savefig(out_png, bbox_inches='tight', dpi=300)
            plt.clf()

            logging.info('Min SNR for B={} is {}'
                         .format(str(curr_shell),
                                 str(np.min(curr_values))))
            logging.info('Max SNR for B={} is {}'
                         .format(str(curr_shell),
                                 str(np.max(curr_values))))
            logging.info('Mean SNR for B={} is {}'
                         .format(str(curr_shell),
                                 str(np.mean(curr_values))))

    else:
        b0_values = df.loc[df['bval'] == 0.0]['snr']

        logging.info('Mean SNR for b0 is {}'.format(str(np.mean(b0_values))))

        curr_values = df.loc[df['bval'] != 0.0]['snr']
        plt.plot(range(len(curr_values)), curr_values,
                 marker='+', linestyle='--')
        plt.legend(["SNR"])
        plt.xlabel("Volume (excluding B0)")
        plt.ylabel("Estimated SNR")
        plt.xlim([-1, len(df)])
        plt.text(1, 9, 'Min SNR B0 = ' +
                 str(np.min(df.loc[df['bval'] == 0.0]['snr'])))
        plt.text(1, 5, 'Max SNR B0 = ' +
                 str(np.max(df.loc[df['bval'] == 0.0]['snr'])))
        plt.text(1, 1, 'Mean SNR B0 = ' +
                 str(np.mean(df.loc[df['bval'] == 0.0]['snr'])))
        plt.savefig(basename + "_graph.png", bbox_inches='tight', dpi=300)
        plt.close()

    min_value = df[df['snr'] == np.min(df['snr'])].index[0]
    max_value = df[df['snr'] == np.max(df['snr'])].index[0]
    logging.info('Min SNR is {} and from B={}'
                 .format(str(df['snr'][min_value]),
                         str(df['bval'][min_value])))
    logging.info('Max SNR is {} and from B={}'
                 .format(str(df['snr'][max_value]),
                         str(df['bval'][max_value])))

    with open(basename + "_SNR.json", "w") as f:
        df.T.to_json(f, indent=args.indent)


if __name__ == "__main__":
    main()
