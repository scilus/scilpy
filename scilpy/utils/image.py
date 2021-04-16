# -*- coding: utf-8 -*-

import os
import logging

from datetime import datetime
from dipy.align.imaffine import (AffineMap,
                                 AffineRegistration,
                                 MutualInformationMetric,
                                 transform_centers_of_mass)
from dipy.align.transforms import (AffineTransform3D,
                                   RigidTransform3D)
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.utils import get_reference_info
from dipy.segment.mask import median_otsu
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scipy.ndimage.morphology import binary_dilation
from scilpy.io.image import get_data_as_mask
from scilpy.utils.filenames import split_name_with_nii


def transform_anatomy(transfo, reference, moving, filename_to_save,
                      interp='linear'):
    """
    Apply transformation to an image using Dipy's tool

    Parameters
    ----------
    transfo: numpy.ndarray
        Transformation matrix to be applied
    reference: str
        Filename of the reference image (target)
    moving: str
        Filename of the moving image
    filename_to_save: str
        Filename of the output image
    interp : string, either 'linear' or 'nearest'
        the type of interpolation to be used, either 'linear'
        (for k-linear interpolation) or 'nearest' for nearest neighbor
    """
    grid2world, dim, _, _ = get_reference_info(reference)
    static_data = nib.load(reference).get_fdata(dtype=np.float32)

    nib_file = nib.load(moving)
    moving_data = nib_file.get_fdata(dtype=np.float32)
    moving_affine = nib_file.affine

    if moving_data.ndim == 3 and isinstance(moving_data[0, 0, 0],
                                            np.ScalarType):
        orig_type = moving_data.dtype
        affine_map = AffineMap(np.linalg.inv(transfo),
                               dim, grid2world,
                               moving_data.shape, moving_affine)
        resampled = affine_map.transform(moving_data.astype(np.float64),
                                         interpolation=interp)
        nib.save(nib.Nifti1Image(resampled.astype(orig_type), grid2world),
                 filename_to_save)
    elif len(moving_data[0, 0, 0]) > 1:
        if isinstance(moving_data[0, 0, 0], np.void):
            raise ValueError('Does not support TrackVis RGB')

        affine_map = AffineMap(np.linalg.inv(transfo),
                               dim[0:3], grid2world,
                               moving_data.shape[0:3], moving_affine)

        orig_type = moving_data.dtype
        resampled = transform_dwi(affine_map, static_data, moving_data,
                                  interpolation=interp)
        nib.save(nib.Nifti1Image(resampled.astype(orig_type), grid2world),
                 filename_to_save)
    else:
        raise ValueError('Does not support this dataset (shape, type, etc)')


def transform_dwi(reg_obj, static, dwi, interpolation='linear'):
    """
    Iteratively apply transformation to 4D image using Dipy's tool

    Parameters
    ----------
    reg_obj: AffineMap
        Registration object from Dipy returned by AffineMap
    static: numpy.ndarray
        Target image data
    dwi: numpy.ndarray
        4D numpy array containing a scalar in each voxel (moving image data)
    interpolation : string, either 'linear' or 'nearest'
        the type of interpolation to be used, either 'linear'
        (for k-linear interpolation) or 'nearest' for nearest neighbor
    """
    trans_dwi = np.zeros(static.shape + (dwi.shape[3],), dtype=dwi.dtype)
    for i in range(dwi.shape[3]):
        trans_dwi[..., i] = reg_obj.transform(dwi[..., i],
                                              interpolation=interpolation)

    return trans_dwi


def register_image(static, static_grid2world, moving, moving_grid2world,
                   transformation_type='affine', dwi=None):
    if transformation_type not in ['rigid', 'affine']:
        raise ValueError('Transformation type not available in Dipy')

    # Set all parameters for registration
    nbins = 32
    params0 = None
    sampling_prop = None
    level_iters = [50, 25, 5]
    sigmas = [8.0, 4.0, 2.0]
    factors = [8, 4, 2]
    metric = MutualInformationMetric(nbins, sampling_prop)
    reg_obj = AffineRegistration(metric=metric, level_iters=level_iters,
                                 sigmas=sigmas, factors=factors, verbosity=0)

    # First, align the center of mass of both volume
    c_of_mass = transform_centers_of_mass(static, static_grid2world,
                                          moving, moving_grid2world)
    # Then, rigid transformation (translation + rotation)
    transform = RigidTransform3D()
    rigid = reg_obj.optimize(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine=c_of_mass.affine)

    if transformation_type == 'affine':
        # Finally, affine transformation (translation + rotation + scaling)
        transform = AffineTransform3D()
        affine = reg_obj.optimize(static, moving, transform, params0,
                                  static_grid2world, moving_grid2world,
                                  starting_affine=rigid.affine)

        mapper = affine
        transformation = affine.affine
    else:
        mapper = rigid
        transformation = rigid.affine

    if dwi is not None:
        trans_dwi = transform_dwi(mapper, static, dwi)
        return trans_dwi, transformation
    else:
        return mapper.transform(moving), transformation


def compute_snr(dwi, bval, bvec, b0_thr, mask,
                noise_mask=None, basename=None, verbose=False):
    """
    Compute snr

    Parameters
    ----------
    dwi: string
        Path to the dwi file
    bvec: string
        Path to the bvec file
    bval: string
        Path to the bval file
    b0_thr: int
        Threshold to define b0 minimum value
    mask: string
        Path to the mask
    noise_mask: string
        Path to the noise mask
    basename: string
        Basename used for naming all output files
    verbose: boolean
        Set to use logging
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    img = nib.load(dwi)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    mask = get_data_as_mask(nib.load(mask), dtype=bool)

    if basename is None:
        filename, ext = split_name_with_nii(dwi)
    else:
        filename = basename

    logging.info('Basename: {}'.format(filename))
    bvals, bvecs = read_bvals_bvecs(bval, bvec)
    b0s_location = bvals <= b0_thr

    if noise_mask is None:
        b0_mask, noise_mask = median_otsu(data, vol_idx=b0s_location)

        # we inflate the mask, then invert it to recover only the noise
        noise_mask = binary_dilation(noise_mask, iterations=10).squeeze()

        # Add the upper half in order to delete the neck and shoulder
        # when inverting the mask
        noise_mask[..., :noise_mask.shape[-1]//2] = 1

        # Reverse the mask to get only noise
        noise_mask = (~noise_mask).astype('float32')

        logging.info('Number of voxels found '
                     'in noise mask : {}'.format(np.count_nonzero(noise_mask)))
        logging.info('Total number of voxel '
                     'in volume : {}'.format(np.size(noise_mask)))

        nib.save(nib.Nifti1Image(noise_mask, affine),
                 filename + '_noise_mask.nii.gz')

    else:
        noise_mask = nib.load(noise_mask).get_data().squeeze()

    signal_mean = np.zeros(data.shape[-1])
    noise_std = np.zeros(data.shape[-1])
    SNR = np.zeros(data.shape[-1])

    # Write everything that is printed to a txt file
    report = open(filename + '_info.txt', 'a')
    report_header = '\n\n--------------------------------------------\n'
    report_header += 'Now beginning processing of {} image '\
                     'at: {}'.format(os.path.dirname(os.path.realpath(dwi)),
                                     str(datetime.now()))
    report_header += '\n\n\n---------------------------------------------\n\n'
    report.write(report_header)

    for idx in range(data.shape[-1]):
        signal_mean[idx] = np.mean(data[..., idx:idx+1][mask > 0])
        noise_std[idx] = np.std(data[..., idx:idx+1][noise_mask > 0])
        SNR[idx] = signal_mean[idx] / noise_std[idx]

        message = '\nNow processing image {} '\
                  'of {}'.format(str(idx),
                                 str(data.shape[-1]-1))
        message += '\nSignal mean is {}\n'.format(str(signal_mean[idx]))
        message += 'Noise Standard deviation is {}'.format(str(noise_std[idx]))
        message += '\nEstimated SNR is {}'.format(str(SNR[idx]))
        message += '\nGradient direction is {}\n'.format(str(bvecs[idx, :]))
        logging.info(message)
        report.write(message)

    SNR_b0 = SNR[b0s_location]
    report_SNR_b0 = '\nSNR for b0 is {}\n'.format(str(SNR_b0))
    report_SNR = 'Max SNR (located at gradient direction {} ) '\
                 'is {}\n'.format(str(bvecs[:, np.argmax(SNR)]),
                                  str(np.max(SNR)))
    report_SNR += 'Min SNR (located at gradient direction {} ) '\
                  'is {}'.format(str(bvecs[:, np.argmin(SNR)]),
                                 str(np.min(SNR)))

    logging.info(report_SNR_b0 + report_SNR)
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
