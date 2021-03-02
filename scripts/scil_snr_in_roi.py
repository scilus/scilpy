#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to signal to noise ratio (SNR) in a region of interest (ROI)

Computes the SNR for all slices of the input image.
The mean of the signal is computed inside the mask.
The standard deviation of the noise is estimated inside noise_mask.
If it's not supplied, it will be estimated using the data outside medotsu.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import os
import argparse

from datetime import datetime
from dipy.segment.mask import median_otsu
from scipy.ndimage.morphology import binary_dilation


def compute_snr(image, bvecs_file, mask, noise_mask=None, savename=None, verbose=False):
    """
    Computes the SNR for all slices of image.
    The mean of the signal is computed inside the mask.
    The standard deviation of the noise is estimated inside noise_mask.
    If it's not supplied, it will be estimated using the data outside medotsu.
    """

    img = nib.load(image)
    mask = nib.load(mask).get_data()

    data = img.get_data()
    header = img.get_header()
    affine = img.get_affine()

    if savename is None:
        temp, _ = str.split(os.path.basename(image), '.', 1)
        filename = os.path.dirname(os.path.realpath(image)) + '/' + temp

    else:
        filename, _ = str.split(savename, '.', 1)

    bvecs = np.loadtxt(bvecs_file)
    b0s_location = np.where(np.sum(np.abs(bvecs), axis=0)==0)

    if noise_mask is None:

        b0_mask, noise_mask = median_otsu(data, b0Slices=b0s_location)

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
    bvecs = np.delete(bvecs, b0s_location, axis=1)
    SNR = np.delete(SNR, b0s_location)
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
    Estimates the SNR in a given region of the anatomy.

    Input : A DWI volume in which to compute the SNR.

            A nifti file image that is a binary mask of the region of interest.

            The bvecs file associated with the DWI.


    This works best by using the output from the corpus callosum segmentation script (segment_CC.py).

    Optional input : a binary mask for the noise region (otherwise it computes one automatically)
    """


def buildArgsParser():

    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('input', action='store', metavar='input',
                   help='Path of the image file.')

    p.add_argument('bvecs', action='store', metavar='bvecs',
                   help='bvecs file (Mrtrix format) used to specify the gradient directions in the report.')

    p.add_argument('mask_roi', action='store', metavar='mask_roi',
                   help='Binary mask of the region used to estimate SNR.')

    p.add_argument('--noise', action='store', dest='noise_mask',
                   metavar='noise_mask',
                   help='Binary mask used to estimate the noise.')

    p.add_argument('-v', action='store_true', dest='verbose',
                   help='If True, prints the information that gets saved into the report.')

    p.add_argument('-o', action='store', dest='savename',
                   metavar='savename', type=str,
                   help='Path and prefix for the various saved file.')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    compute_snr(args.input, args.bvecs, args.mask_roi, args.noise_mask, args.savename, args.verbose)


if __name__ == "__main__":
    main()
