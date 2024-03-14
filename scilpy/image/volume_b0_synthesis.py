# -*- coding: utf-8 -*-
import logging
import os
import warnings

# Disable tensorflow warnings
with warnings.catch_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.simplefilter("ignore")
    from dipy.nn.synb0 import Synb0

import numpy as np
from dipy.align.imaffine import AffineMap
from dipy.segment.tissue import TissueClassifierHMRF
from scipy.ndimage import gaussian_filter

from scilpy.image.volume_operations import register_image


def compute_b0_synthesis(t1_data, t1_bet_data, b0_data, b0_bet_data,
                         template_data, t1_affine, b0_affine, template_affine,
                         verbose):
    """
    Note. Tensorflow is required here, through dipy.Synb0. If not installed,
    dipy will raise an error, like:
    >> dipy.utils.tripwire.TripWireError: We need package tensorflow_addons for
    these functions, but ``import tensorflow_addons`` raised an ImportError.

    Parameters
    ----------
    t1_data: np.ndarary
        The T1 (wholebrain, with the skull).
    t1_bet_data: np.ndarray
        The mask.
    b0_data: np.ndarray
        The b0 (wholebrain, with the skull).
    b0_bet_data: np.ndarray
        The mask.
    template_data: np.ndarray
        The template for typical usage.
    t1_affine: np.ndarray
        The T1's affine.
    b0_affine: np.ndarray
        The b0's affine.
    template_affine: np.ndarray
        The template's affine
    verbose: bool
        Whether to make dipy's Synb0 verbose.

    Returns
    -------
    rev_b0: np.ndarray
        The synthetized b0.
    """
    logging.info('The usage of synthetic b0 is not fully tested.'
                 'Be careful when using it.')

    # Crude estimation of the WM mean intensity and normalization
    logging.info('Estimating WM mean intensity')
    hmrf = TissueClassifierHMRF()
    t1_bet_data = gaussian_filter(t1_bet_data, 2)
    _, final_segmentation, _ = hmrf.classify(t1_bet_data, 3, 0.25,
                                             tolerance=1e-4, max_iter=5)
    avg_wm = np.mean(t1_data[final_segmentation == 3])

    # Modifying t1
    t1_data /= avg_wm
    t1_data *= 110

    # SyNB0 works only in a standard space, so we need to register the images
    logging.info('Registering images')
    # Use the BET image for registration
    t1_bet_to_b0, t1_bet_to_b0_transform = register_image(
        b0_bet_data, b0_affine, t1_bet_data, t1_affine, fine=True)
    affine_map = AffineMap(t1_bet_to_b0_transform,
                           b0_data.shape, b0_affine,
                           t1_data.shape, t1_affine)
    t1_skull_to_b0 = affine_map.transform(t1_data.astype(np.float64))

    # Then register to MNI (using the BET again)
    _, t1_bet_to_b0_to_mni_transform = register_image(
        template_data, template_affine, t1_bet_to_b0, b0_affine, fine=True)
    affine_map = AffineMap(t1_bet_to_b0_to_mni_transform,
                           template_data.shape, template_affine,
                           b0_data.shape, b0_affine)

    # But for prediction, we want the skull
    b0_skull_to_mni = affine_map.transform(b0_data.astype(np.float64))
    t1_skull_to_mni = affine_map.transform(t1_skull_to_b0.astype(np.float64))

    logging.info('Running SyN-B0')
    SyNb0 = Synb0(verbose)
    rev_b0 = SyNb0.predict(b0_skull_to_mni, t1_skull_to_mni)
    rev_b0 = affine_map.transform_inverse(rev_b0.astype(np.float64))

    return rev_b0
