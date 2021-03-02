#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute signal to noise ratio (SNR) in a region of interest (ROI)
of a DWI volume. 

It will compute the SNR for all DWI volumes of the input image seperately.
The output will contain the SNR 
The mean of the signal is computed inside the mask.
The standard deviation of the noise is estimated inside noise_mask.
If it's not supplied, it will be estimated using the data outside medotsu.

If verbose is True, the SNR for every DWI volume will be outputed.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import os
import argparse

from datetime import datetime
from dipy.segment.mask import median_otsu
from dipy.io.gradients import read_bvals_bvecs
from scipy.ndimage.morphology import binary_dilation
from scilpy.io.image import get_data_as_mask

def compute_snr(image, bvals_file, bvecs_file, b0_thr,
                mask, noise_mask=None, savename=None, verbose=False):

    img = nib.load(image)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    mask = get_data_as_mask(nib.load(mask), dtype=bool)

    if savename is None:
        temp, _ = str.split(os.path.basename(image), '.', 1)
        filename = os.path.dirname(os.path.realpath(image)) + '/' + temp
    else:
        filename, _ = str.split(savename, '.', 1)

    bvals, bvecs = read_bvals_bvecs(bvals_file, bvecs_file)
    b0s_location = bvals <= b0_thr
    bvecs = np.loadtxt(bvecs_file)

    if noise_mask is None:
        b0_mask, noise_mask = median_otsu(data, vol_idx=b0s_location)

        # we inflate the mask, then invert it to recover only the noise
        noise_mask = binary_dilation(noise_mask, iterations=10).squeeze()

        # Add the upper half in order to delete the neck and shoulder
        # when inverting the mask
        noise_mask[..., :noise_mask.shape[-1]//2] = 1

        # Reverse the mask to get only noise
        noise_mask = (~noise_mask).astype('float32')

        if verbose:
            print("Number of voxels found in noise mask :", np.count_nonzero(noise_mask), \
                  "Total number of voxel in volume :", np.size(noise_mask))

        nib.save(nib.Nifti1Image(noise_mask, affine), filename + '_noise_mask.nii.gz')

    else:
        noise_mask = nib.load(noise_mask).get_data().squeeze()

    signal_mean = np.zeros(data.shape[-1])
    noise_std = np.zeros(data.shape[-1])
    SNR = np.zeros(data.shape[-1])

    # Write everything that is printed to a txt file
    report = open(filename + '_info.txt', 'a')
    report_header = "\n\n--------------------------------------------\nNow beginning processing of " + \
                    os.path.dirname(os.path.realpath(image)) + image + " at : " + str(datetime.now()) + \
                    "\n---------------------------------------------\n\n"
    report.write(report_header)

    for idx in range(data.shape[-1]):
        signal_mean[idx] = np.mean(data[..., idx:idx+1][mask > 0])
        noise_std[idx] = np.std(data[..., idx:idx+1][noise_mask > 0])
        SNR[idx] = signal_mean[idx]/noise_std[idx]

        info = "\nNow processing image " + str(idx) + " of " + str(data.shape[-1]-1) + '\n' + "Signal mean is : " + str(signal_mean[idx]) + \
               "\n" + "Noise Standard deviation is : " + str(noise_std[idx]) + \
               "\nEstimated SNR is : " + str(SNR[idx]) + "\nGradient direction is " + str(bvecs[:, idx]) + "\n"
        if verbose:
            print (info)

        report.write(info)

    SNR_b0 = SNR[b0s_location]
    report_SNR_b0 = "\nSNR for b0 is " + str(SNR_b0)
    report_SNR = "\nMax SNR (located at gradient direction " + \
                str(bvecs[:, np.argmax(SNR)]) +  " ) is : " + str(np.max(SNR)) + \
                "\nMin SNR (located at gradient direction " + \
                str(bvecs[:, np.argmin(SNR)]) + ") is : " + str(np.min(SNR))

    print (report_SNR_b0 + report_SNR)
    report.write(report_SNR_b0 + report_SNR)
    report.close()

    plt.plot(SNR)
    plt.legend(["SNR"])
    plt.xlabel("Volume (excluding B0)")
    plt.ylabel("Estimated SNR")
    plt.text(1, 1, 'SNR B0 = ' + str(SNR_b0))

    plt.savefig(filename + "_graph_SNR.png", bbox_inches='tight', dpi=300)

    # Save the numpy arrays used for SNR calculation
    np.save(filename + '_std', noise_std)
    np.save(filename + '_mean', signal_mean)
    np.save(filename + '_SNR', SNR)


DESCRIPTION = """
    Estimates the SNR in a given region of interest (ROI).
    SNR is the ratio of the average signal in the ROI divide by standard deviation.

    This works best in a well-defined ROI such as the corpus callosum. 
    It is heavily dependent on the ROI and its quality. 

    Optional input : a binary mask for the noise region (otherwise it computes one automatically)
    """


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('in_dwi', 
                   help='Path of the input diffusion volume.')

    p.add_argument('in_bval',
                   help='Path of the bvals file, in FSL format.')

    p.add_argument('in_bvec',
                   help='Path of the bvecs file, in FSL format.')

    p.add_argument('mask_roi', action='store', metavar='mask_roi',
                   help='Binary mask of the region used to estimate SNR.')

    p.add_argument('--noise', action='store', dest='noise_mask',
                   metavar='noise_mask',
                   help='Binary mask used to estimate the noise.')

    p.add_argument('--b0_thr', type=float, default=0.0,
                   help='All b-values with values less than or equal '
                        'to b0_thr are considered as b0s i.e. without '
                        'diffusion weighting. [Default: 0.0]')

    p.add_argument('-v', action='store_true', dest='verbose',
                   help='If True, prints the information that gets saved into the report.')

    p.add_argument('-o', action='store', dest='savename',
                   metavar='savename', type=str,
                   help='Path and prefix for the various saved file.')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    compute_snr(args.in_dwi, args.in_bval, args.in_bvec, args.b0_thr,
                args.mask_roi, args.noise_mask, args.savename, args.verbose)


if __name__ == "__main__":
    main()
