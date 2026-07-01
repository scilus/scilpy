# -*- coding:utf-8 -*-
"""
[Note - 2025/11/14]
Code for Frangi filters adapted from https://github.com/lens-biophotonics/Foa3D.

The original MIT license is copied below.

================================
MIT License

Copyright (c) 2022 lens-biophotonics

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

================================
[Note from the original authors]

Portions of code adapted from https://github.com/ellisdg/frangi3d (MIT license)
"""

from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi


def divide_nonzero(nd_array1, nd_array2, new_val=1e-10):
    """
    Divide two arrays handling zero denominator values.

    Parameters
    ----------
    nd_array1: numpy.ndarray
        dividend array

    nd_array2: numpy.ndarray
        divisor array

    new_val: float
        substituted value

    Returns
    -------
    divided: numpy.ndarray
        divided array
    """
    divisor = np.copy(nd_array2)
    divisor[divisor == 0] = new_val
    divided = np.divide(nd_array1, divisor)

    return divided


def analyze_hessian_eigen(img, sigma, trunc=4,
                          padding_mode='reflect',
                          padding_cval=0):
    """
    Compute the eigenvalues of local Hessian matrices
    of the input image array, sorted by absolute value (in ascending order),
    along with the related (dominant) eigenvectors.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    sigma: int
        spatial scale [px]

    trunc: int
        truncate the Gaussian smoothing kernel at this many standard deviations

    Returns
    -------
    eigval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)

    dom_eigvec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvectors related to the dominant (minimum) eigenvalue
    """
    hessian = compute_scaled_hessian(img, sigma=sigma, trunc=trunc,
                                     padding_mode=padding_mode, padding_cval=padding_cval)
    eigval, dom_eigvec = compute_dominant_eigen(hessian)

    return eigval, dom_eigvec


def compute_dominant_eigen(hessian):
    """
    Compute the eigenvalues (sorted by absolute value)
    of symmetrical Hessian matrix, selecting the eigenvectors related to the dominant ones.

    Parameters
    ----------
    hessian: numpy.ndarray (axis order=(Z,Y,X,C,C), dtype=float)
        input array of local Hessian matrices

    Returns
    -------
    srt_eigval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)

    dom_eigvec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvectors related to the dominant (minimum) eigenvalue
    """
    eigenval, eigenvec = np.linalg.eigh(hessian)
    srt_eigval, srt_eigvec = sort_eigen(eigenval, eigenvec)
    dom_eigvec = srt_eigvec[..., 0]

    return srt_eigval, dom_eigvec


def compute_frangi_features(eigen1, eigen2, eigen3, gamma):
    """
    Compute the basic image features employed by the Frangi filter.

    Parameters
    ----------
    eigen1: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        lowest Hessian eigenvalue (i.e., the dominant eigenvalue)

    eigen2: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    gamma: float
        background score sensitivity

    Returns
    -------
    ra: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        plate-like object score

    rb: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        blob-like object score

    s: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
       second-order structureness

    gamma: float
        background score sensitivity
        (automatically computed if not provided as input)
    """
    ra = divide_nonzero(np.abs(eigen2), np.abs(eigen3))
    rb = divide_nonzero(np.abs(eigen1), np.sqrt(np.abs(np.multiply(eigen2, eigen3))))
    s = compute_structureness(eigen1, eigen2, eigen3)

    # compute default gamma sensitivity
    if gamma is None:
        gamma = 0.5 * np.max(s)

    return ra, rb, s, gamma


def compute_scaled_hessian(img, sigma=1, trunc=4, padding_mode='reflect', padding_cval=0):
    """
    Computes the scaled and normalized Hessian matrices of the input image.
    This is then used to estimate Frangi's vesselness probability score.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image

    sigma: int
        spatial scale [px]

    trunc: int
        truncate the Gaussian smoothing kernel at this many standard deviations

    padding_mode: str
        mode for padding the image

    padding_cval: float
        constant value for padding

    Returns
    -------
    hessian: numpy.ndarray (axis order=(Z,Y,X,C,C), dtype=float)
        Hessian matrix of image second derivatives
    """
    # scale selection
    scaled_img = ndi.gaussian_filter(img, sigma=sigma, output=np.float32,
                                     truncate=trunc,
                                     mode=padding_mode if padding_mode != 'edge' else 'nearest',
                                     cval=padding_cval)

    # compute the first order gradients
    gradient_list = np.gradient(scaled_img)

    # compute the Hessian matrix elements
    hessian_elements = [np.gradient(gradient_list[ax0], axis=ax1)
                        for ax0, ax1 in combinations_with_replacement(range(img.ndim), 2)]

    # scale the elements of the Hessian matrix
    corr_factor = sigma ** 2
    hessian_elements = [corr_factor * element for element in hessian_elements]

    # create the Hessian matrix from its basic elements
    hessian = np.zeros((img.ndim, img.ndim) + scaled_img.shape, dtype=scaled_img.dtype)
    for index, (ax0, ax1) in enumerate(combinations_with_replacement(range(img.ndim), 2)):
        element = hessian_elements[index]
        hessian[ax0, ax1, ...] = element
        if ax0 != ax1:
            hessian[ax1, ax0, ...] = element

    # re-arrange axes
    hessian = np.moveaxis(hessian, (0, 1), (-2, -1))

    return hessian


def compute_plate_like_score(ra, alpha):
    return 1 - np.exp(np.divide(np.negative(np.square(ra)), 2 * np.square(alpha)))


def compute_blob_like_score(rb, beta):
    return np.exp(np.divide(np.negative(np.square(rb)), 2 * np.square(beta)))


def compute_background_score(s, gamma):
    return 1 - np.exp(np.divide(np.negative(np.square(s)), 2 * np.square(gamma)))


def compute_structureness(eigen1, eigen2, eigen3):
    return np.sqrt(np.square(eigen1) + np.square(eigen2) + np.square(eigen3))


def compute_scaled_vesselness(eigen1, eigen2, eigen3, alpha, beta, gamma):
    """
    Estimate Frangi's vesselness probability.

    Parameters
    ----------
    eigen1: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        lowest Hessian eigenvalue (i.e., the dominant eigenvalue)

    eigen2: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    alpha: float
        plate-like score sensitivity

    beta: float
        blob-like score sensitivity

    gamma: float
        background score sensitivity

    Returns
    -------
    vesselness: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood image
    """
    ra, rb, s, gamma = compute_frangi_features(eigen1, eigen2, eigen3, gamma)
    plate = compute_plate_like_score(ra, alpha)
    blob = compute_blob_like_score(rb, beta)
    background = compute_background_score(s, gamma)
    vesselness = plate * blob * background

    return vesselness


def compute_scaled_orientation(scale_px, img, alpha=0.001, beta=1, gamma=None,
                               padding_mode='reflect', padding_cval=0):
    """
    Compute fiber orientation vectors at the input spatial scale of interest

    Parameters
    ----------
    scale_px: int
        spatial scale [px]
    img: numpy.ndarray (axis order=(Z,Y,X))
        3D microscopy image
    alpha: float
        plate-like score sensitivity
    beta: float
        blob-like score sensitivity
    gamma: float
        background score sensitivity
    padding_mode: str
        mode for padding the image
    padding_cval: float
        constant value for padding

    Returns
    -------
    frangi_img: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        Frangi's vesselness likelihood image
    eigvec: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        3D orientation map at the input spatial scale
    eigval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        Hessian eigenvalues sorted by absolute value (ascending order)
    """
    # compute local Hessian matrices and perform eigenvalue decomposition
    eigval, eigvec = analyze_hessian_eigen(img, scale_px, trunc=4,
                                           padding_mode=padding_mode,
                                           padding_cval=padding_cval)

    # compute Frangi's vesselness probability image
    eigen1, eigen2, eigen3 = eigval
    vesselness = compute_scaled_vesselness(eigen1, eigen2, eigen3, alpha=alpha, beta=beta, gamma=gamma)
    frangi_img = reject_vesselness_background(vesselness, eigen2, eigen3)
    eigval = np.stack(eigval, axis=-1)

    return frangi_img, eigvec, eigval


def frangi_filter(img, scales_px=1, *, alpha=0.001, beta=1.0, gamma=None, threshold=1e-4,
                  padding_mode='reflect', padding_cval=0):
    """
    Apply 3D Frangi filter to 3D microscopy image.

    Parameters
    ----------
    img: numpy.ndarray (axis order=(X,Y,Z))
        3D microscopy image
    scales_px: int or list (dtype=int)
        analyzed spatial scales [px]
    alpha: float
        plate-like score sensitivity
    beta: float
        blob-like score sensitivity
    gamma: float
        background score sensitivity (if None, gamma is automatically tailored)
    threshold: float
        vesselness threshold for accepting a direction
    mask_eigen: str = None
        whether to mask positive eigenvalues for vesselness measure
    padding_mode: str
        mode for padding the image
    padding_cval: float
        constant value for padding

    Returns
    -------
    frangi: numpy.ndarray (axis order=(X,Y,Z), dtype=float)
        Frangi's vesselness likelihood image
    vec: numpy.ndarray (axis order=(X,Y,Z,C), dtype=float)
        3D fiber orientation field
    """
    # single-scale or parallel multi-scale vesselness analysis
    ns = len(scales_px)

    frangi_max = np.full(img.shape, threshold, dtype='float32')
    eigvec_max = np.zeros(img.shape + (3,), dtype='float32')
    for s in range(ns):
        frangi, eigvec, _ = compute_scaled_orientation(scales_px[s], img, alpha=alpha,
                                                       beta=beta, gamma=gamma,
                                                       padding_mode=padding_mode,
                                                       padding_cval=padding_cval)
        max_where = frangi_max < frangi
        frangi_max[max_where] = frangi[max_where]
        eigvec_max[max_where] = eigvec[max_where]

    frangi = frangi_max
    fbr_vec = eigvec_max

    return frangi, fbr_vec


def reject_vesselness_background(vesselness, eigen2, eigen3):
    """
    Reject the fiber background, exploiting the sign of the "secondary"
    eigenvalues λ2 and λ3.

    Parameters
    ----------
    vesselness: numpy.ndarray (axis order=(Z,Y,X))
        Frangi's vesselness likelihood image

    eigen2: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        middle Hessian eigenvalue

    eigen3: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        highest Hessian eigenvalue

    Returns
    -------
    vesselness: numpy.ndarray (axis order=(Z,Y,X), dtype=float)
        masked Frangi's vesselness likelihood image
    """
    bg_msk = np.isnan(vesselness)
    bg_msk = np.logical_or(bg_msk, np.logical_or(eigen2 > 0, eigen3 > 0))

    # set background vesselness to zero
    vesselness[bg_msk] = 0
    return vesselness


def sort_eigen(eigval, eigvec, axis=-1):
    """
    Sort eigenvalue and related eigenvector arrays
    by absolute value along the given axis.

    Parameters
    ----------
    eigval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        original eigenvalue array

    eigvec: numpy.ndarray (axis order=(Z,Y,X,C,C), dtype=float)
        original eigenvector array

    axis: int
        sorted axis

    Returns
    -------
    srt_eigval: numpy.ndarray (axis order=(Z,Y,X,C), dtype=float)
        sorted eigenvalue array (ascending order)

    srt_eigvec: numpy.ndarray (axis order=(Z,Y,X,C,C), dtype=float)
        sorted eigenvector array
    """
    # sort the eigenvalue array by absolute value (ascending order)
    srt_val_idx = np.abs(eigval).argsort(axis)
    srt_eigval = np.take_along_axis(eigval, srt_val_idx, axis)
    srt_eigval = [np.squeeze(eigval, axis=axis) for eigval in np.split(srt_eigval, srt_eigval.shape[axis], axis=axis)]

    # sort related eigenvectors consistently
    srt_vec_idx = srt_val_idx[:, :, :, np.newaxis, :]
    srt_eigvec = np.take_along_axis(eigvec, srt_vec_idx, axis)

    return srt_eigval, srt_eigvec
