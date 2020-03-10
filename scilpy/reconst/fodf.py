# !/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import logging

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction.peaks import peaks_from_model

from scilpy.utils.bvec_bval_tools import normalize_bvecs, is_normalized_bvecs
from scilpy.io.utils import check_sh_basis_choice
from scilpy.utils.bvec_bval_tools import check_b0_threshold


def compute_fodf(data, bvals, bvecs, full_frf, sh_order=8, nbr_processes=None,
                 mask=None, sh_basis='descoteaux07', return_sh=True,
                 n_peaks=5, force_b0_threshold=False):
    """
     Script to compute Constrained Spherical Deconvolution (CSD) fiber ODFs.

     By default, will output all possible files, using default names. Specific
     names can be specified using the file flags specified in the "File flags"
     section.

    If --not_all is set, only the files specified explicitly by the flags
    will be output.

    See [Tournier et al. NeuroImage 2007] and [Cote et al Tractometer MedIA 2013]
    for quantitative comparisons with Sharpening Deconvolution Transform (SDT).

    Parameters
    ----------
    data: ndarray
        4D Input diffusion volume with shape (X, Y, Z, N)
    bvals: ndarray
        1D bvals array with shape (N,)
    bvecs: ndarray
        2D (normalized) bvecs array with shape (N, 3)
    full_frf: ndarray
        frf data, ex, loaded from a frf_file, with shape (4,).
    sh_order: int, optional
        SH order used for the CSD. (Default: 8)
    nbr_processes: int, optional
        Number of sub processes to start. Default = none, i.e use the cpu count.
    mask: ndarray, optional
        3D mask with shape (X,Y,Z)
        Binary mask. Only the data inside the mask will be used for
        computations and reconstruction. Useful if no white matter mask is
        available.
    sh_basis: str, optional
        Spherical harmonics basis used for the SH coefficients.Must be either
        'descoteaux07' or 'tournier07' (default 'descoteaux07')
        - 'descoteaux07': SH basis from the Descoteaux et al. MRM 2007 paper
        - 'tournier07': SH basis from the Tournier et al. NeuroImage 2007 paper.
    return_sh: bool, optional
        If true, returns the sh.
    n_peaks: int, optional
        Nb of peaks for the fodf. Default: copied dipy's default, i.e. 5.
    force_b0_threshold: bool, optional
        If True, will continue even if the minimum bvalue is suspiciously high.

    Returns
    -------
    peaks_csd: PeaksAndMetrics
        An object with ``gfa``, ``peak_directions``, ``peak_values``,
        ``peak_indices``, ``odf``, ``shm_coeffs`` as attributes
    """

    # Checking data and sh_order
    check_b0_threshold(force_b0_threshold, bvals.min())
    if data.shape[-1] < (sh_order + 1) * (sh_order + 2) / 2:
        warnings.warn(
            'We recommend having at least {} unique DWI volumes, but you '
            'currently have {} volumes. Try lowering the parameter sh_order '
            'in case of non convergence.'.format(
                (sh_order + 1) * (sh_order + 2) / 2, data.shape[-1]))

    # Checking bvals, bvecs values and loading gtab
    if not is_normalized_bvecs(bvecs):
        logging.warning('Your b-vectors do not seem normalized...')
        bvecs = normalize_bvecs(bvecs)
    gtab = gradient_table(bvals, bvecs, b0_threshold=bvals.min())

    # Checking full_frf and separating it
    if not full_frf.shape[0] == 4:
        raise ValueError('FRF file did not contain 4 elements. '
                         'Invalid or deprecated FRF format')
    frf = full_frf[0:3]
    mean_b0_val = full_frf[3]

    # Checking if we will use parallel processing
    parallel = True
    if nbr_processes is not None:
        if nbr_processes <= 0:
            nbr_processes = None        # C'est ok ça?? parallel reste true si nbr_processes <0? Je laisse comme c'était.
        elif nbr_processes == 1:
            parallel = False

    # Checking sh basis
    check_sh_basis_choice(sh_basis)

    # Loading the spheres
    reg_sphere = get_sphere('symmetric362')
    peaks_sphere = get_sphere('symmetric724')

    # Computing CSD
    csd_model = ConstrainedSphericalDeconvModel(
        gtab, (frf, mean_b0_val),
        reg_sphere=reg_sphere,
        sh_order=sh_order)

    # Computing peaks. Run in parallel, using the default number of processes
    # (default: CPU count)
    peaks_csd = peaks_from_model(model=csd_model,
                                 data=data,
                                 sphere=peaks_sphere,
                                 relative_peak_threshold=.5,
                                 min_separation_angle=25,
                                 mask=mask,
                                 return_sh=return_sh,
                                 sh_basis_type=sh_basis,
                                 sh_order=sh_order,
                                 normalize_peaks=True,
                                 npeaks=n_peaks,
                                 parallel=parallel,
                                 nbr_processes=nbr_processes)

    return peaks_csd
