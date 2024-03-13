# -*- coding: utf-8 -*-

from copy import deepcopy
import logging

from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh
import numpy as np
from scipy.ndimage import gaussian_filter

import scilpy.tractanalysis.todi_util as todi_u

MINIMUM_TODI_EPSILON = 1e-8
GAUSSIAN_TRUNCATE = 2.0


class TrackOrientationDensityImaging(object):
    def __init__(self, img_shape, sphere_type='repulsion724'):
        """Build the TODI object.

        Histogram distribution of streamlines' local orientations (TODI)
        with a Spherical Function (SF).

        Parameters
        ----------
        img_shape : tuple, list, array
            Dimensions of the reference image.
        sphere_type : str
            The distribution of orientation is discretize on that sphere
            (default 'repulsion724').

        Notes
        -----
        Dhollander, Thijs, et al. "Track orientation density imaging (TODI) and
            track orientation distribution (TOD) based tractography."
            NeuroImage 94 (2014): 312-336.
        """
        assert len(img_shape) == 3

        self.sphere = get_sphere(sphere_type)
        self.nb_sphere_vts = len(self.sphere.vertices)

        self.img_shape = img_shape
        self.todi_shape = img_shape + (self.nb_sphere_vts,)
        self.img_dim = len(img_shape)
        self.nb_voxel = np.prod(self.img_shape)

        self.mask = None
        self.todi = None

    def set_todi(self, mask, todi):
        self.mask = mask
        self.todi = todi

    def compute_todi(self, streamlines, length_weights=True,
                     n_steps=1, asymmetric=False):
        """Compute the TODI map.

        At each voxel an histogram distribution of
        the local streamlines orientations (TODI) is computed.

        Parameters
        ----------
        streamlines : list of numpy.ndarray
            List of streamlines.
        length_weights : bool, optional
            Weights TODI map of each segment's length (default True).
        """
        # Streamlines vertices in "VOXEL_SPACE" within "img_shape" range
        pts_pos, pts_dir, pts_norm = \
            todi_u.streamlines_to_pts_dir_norm(streamlines,
                                               n_steps=n_steps,
                                               asymmetric=asymmetric)

        if not length_weights:
            pts_norm = None

        sph_ids = todi_u.get_dir_to_sphere_id(pts_dir, self.sphere.vertices)

        # Get voxel indices for each point (works because voxels
        # are of unit size and streamlines are scaled accordingly)
        pts_unmasked_vox = todi_u.get_indices_1d(self.img_shape, pts_pos)

        # Generate mask from streamlines vertices
        self.mask = \
            todi_u.generate_mask_indices_1d(self.nb_voxel, pts_unmasked_vox)

        mask_vox_lut = np.cumsum(self.mask) - 1
        nb_voxel_with_pts = mask_vox_lut[-1] + 1
        pts_vox = mask_vox_lut[pts_unmasked_vox]

        # Bincount of each direction at each voxel position
        todi_bin_shape = (nb_voxel_with_pts, self.nb_sphere_vts)
        todi_bin_len = np.prod(todi_bin_shape)

        # Count number of direction for each voxel containing streamlines
        todi_bin_1d = np.bincount(
            np.ravel_multi_index(np.stack((pts_vox, sph_ids)), todi_bin_shape),
            weights=pts_norm, minlength=todi_bin_len)

        # Bincount of sphere id for each voxel
        self.todi = todi_bin_1d.reshape(todi_bin_shape)

    def get_todi(self):
        return self.todi

    def get_tdi(self):
        """Compute the TDI map.

        Compute the Tract Density Image (TDI) from the TODI volume.

        Returns
        -------
        tdi : numpy.ndarray (3D)
            Tract Density Image
        """
        return np.sum(self.todi, axis=-1)

    def get_todi_shape(self):
        return self.todi_shape

    def get_mask(self):
        return self.mask

    def mask_todi(self, mask):
        """Mask the TODI map.

        Mask the TODI without having to reshape the whole volume all
        at once (big in memory).

        Parameters
        ----------
        mask : numpy.ndarray
            Given volume mask for the TODI map.
        """
        # Compute intersection between current mask and given mask
        new_mask = np.logical_and(self.mask, mask.flatten())

        # Prepare new todi
        nb_voxel_with_pts = np.count_nonzero(new_mask)
        new_todi = np.zeros((nb_voxel_with_pts, self.nb_sphere_vts))
        # Too big in memory, mask one dir each step
        for i in range(self.nb_sphere_vts):
            new_todi[:, i] = \
                self.reshape_to_3d(self.todi[:, i]).flatten()[new_mask]
        self.mask = new_mask
        self.todi = new_todi

    def smooth_todi_dir(self, order=2):
        """Smooth orientations on the sphere.

        Smooth the orientations / distribution on the sphere.
        Important for priors construction of BST.

        Parameters
        ----------
        order : int, optional
            Exponent blurring factor, based on the dot product
            (default 2).
        """
        assert order >= 1
        todi_sum = np.sum(self.todi, axis=-1, keepdims=True)
        sphere_dot = np.dot(self.sphere.vertices, self.sphere.vertices.T)
        sphere_psf = np.abs(sphere_dot) ** order
        self.todi = np.dot(self.todi, sphere_psf)
        self.todi *= todi_sum / np.sum(self.todi, axis=-1, keepdims=True)

    def smooth_todi_spatial(self, sigma=0.5):
        """Spatial Smoothing of the TODI map.

        Blur the TODI map using neighborhood information.
        Important for priors construction of BST.

        Parameters
        ----------
        sigma : float, optional
            Gaussian blurring factor (default 0.5).
        """
        # This operation changes the mask as well as the TODI
        mask_3d = self.reshape_to_3d(self.mask).astype(float)
        mask_3d = gaussian_filter(
            mask_3d, sigma, truncate=GAUSSIAN_TRUNCATE).flatten()
        new_mask = mask_3d > MINIMUM_TODI_EPSILON

        # Memory friendly version
        chunk_size = 50
        chunk_count = (self.nb_sphere_vts // chunk_size) + 1
        nb_voxel_with_pts = np.count_nonzero(new_mask)
        new_todi = np.array([])
        tmp_todi = np.zeros((nb_voxel_with_pts, chunk_size))
        # To save on hstack, one chunk at the time
        while chunk_count > 0:
            # Smooth one direction at a time, too big in memory otherwise
            for i in range(chunk_size):
                if i > self.todi.shape[1]-1:
                    tmp_todi = np.delete(
                        tmp_todi, range(i, chunk_size), axis=1)
                    break
                current_vol = self.reshape_to_3d(self.todi[:, i])
                tmp_todi[:, i] = gaussian_filter(
                    current_vol, sigma,
                    truncate=GAUSSIAN_TRUNCATE).flatten()[new_mask]
            # The first hstack cannot be with an empty array
            if new_todi.size == 0:
                new_todi = deepcopy(tmp_todi)
            else:
                new_todi = np.hstack((new_todi, tmp_todi))
            self.todi = np.delete(self.todi, range(
                0, min(self.todi.shape[1], chunk_size)), axis=1)
            chunk_count -= 1

        self.mask = new_mask
        self.todi = new_todi

    def normalize_todi_per_voxel(self, p_norm=2):
        """Normalize TODI map.

        Normalize TODI distribution on the sphere for each voxel independently.

        Parameters
        ----------
        p_norm : int, optional
            Chosen Norm to normalize.

        Returns
        -------
        todi : numpy.ndarray
            Normalized TODI map.
        """
        self.todi = todi_u.p_normalize_vectors(self.todi, p_norm)
        return self.todi

    def get_sh(self, sh_basis, sh_order, smooth=0.006, full_basis=False,
               is_legacy=True):
        """Spherical Harmonics (SH) coefficients of the TODI map

        Compute the SH representation of the TODI map,
        converting SF to SH with a smoothing factor.

        Parameters
        ----------
        sh_basis : {None, 'tournier07', 'descoteaux07'}
            ``None`` for the default DIPY basis,
            ``tournier07`` for the Tournier 2007 [2]_ basis, and
            ``descoteaux07`` for the Descoteaux 2007 [1]_ basis
            (``None`` defaults to ``descoteaux07``).
        sh_order : int
            Maximum SH order in the SH fit.  For `sh_order`, there will be
            ``(sh_order + 1) * (sh_order_2) / 2`` SH coefficients (default 4).
        smooth : float, optional
            Smoothing factor for the conversion,
            Lambda-regularization in the SH fit (default 0.006).
        is_legacy : bool, optional
            Whether or not the SH basis is in its legacy form.

        Returns
        -------
        todi_sh : ndarray
            SH representation of the TODI map

        References
        ----------
        .. [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
               Regularized, Fast, and Robust Analytical Q-ball Imaging.
               Magn. Reson. Med. 2007;58:497-510.
        .. [2] Tournier J.D., Calamante F. and Connelly A.
               Robust determination of the fibre orientation distribution in
               diffusion MRI: Non-negativity constrained super-resolved
               spherical deconvolution. NeuroImage. 2007;35(4):1459-1472.
        """
        return sf_to_sh(self.todi, self.sphere, sh_order_max=sh_order,
                        basis_type=sh_basis, full_basis=full_basis,
                        smooth=smooth, legacy=is_legacy)

    def reshape_to_3d(self, img_voxelly_masked):
        """Reshape a complex ravelled image to 3D.

        Unravel a given unravel mask (1D), image (1D), SH/SF (2D)
        to its original 3D shape (with a 4D for SH/SF).

        Parameters
        ----------
        img_voxelly_masked : numpy.ndarray (either in 1D, 2D or 3D)
            That will be reshaped to 3D (input data shape) accordingly.
            Necessary for future masking operation.

        Returns
        -------
        unraveled_img : numpy.ndarray (3D, or 4D)
            Unravel volume in x, y, z (, c).
        """
        dtype = img_voxelly_masked.dtype
        if img_voxelly_masked.ndim == 1:
            if len(img_voxelly_masked) == self.nb_voxel:
                return img_voxelly_masked.reshape(self.img_shape)

            img_unmasked = np.zeros((self.nb_voxel), dtype=dtype)
            img_unmasked[self.mask] = img_voxelly_masked
            return img_unmasked.reshape(self.img_shape)

        elif img_voxelly_masked.ndim == 2:
            img_last_dim_len = img_voxelly_masked.shape[1]
            img_shape = self.img_shape + (img_last_dim_len,)
            img_unmasked = np.zeros(
                (self.nb_voxel, img_last_dim_len), dtype=dtype)

            img_unmasked[self.mask] = img_voxelly_masked
            return np.reshape(img_unmasked, img_shape)

        logging.warning("WARNING : Volume might already be in 3d shape")
        return img_voxelly_masked

    def compute_distance_to_peak(self, peak_img, normalize_count=True,
                                 deg=True, with_avg_dir=True):
        """Compute distance to peak map.

        Compute the distance of the TODI map to peaks at each position,
        in radian or degree.

        Parameters
        ----------
        peak_img : numpy.ndarray (4D)
            Peaks image, as written by most of Scilpy scripts.
        normalize_count : bool, optional
            Normalize/weight the error map by the density map (default True).
        deg : bool, optional
            Returned error map as degree instead of radian (default True).
        with_avg_dir : bool, optional
            Average all orientation of each voxel of the TODI map
            into a single direction, warning for crossing (default True).

        Returns
        -------
        error_map : numpy.ndarray (3D)
            Average angle error map per voxel.
        """
        assert peak_img.shape[-1] == 3
        if peak_img.ndim == 4:
            peak_img = peak_img.reshape((-1, 3))

        peak_img = peak_img[self.mask]

        if with_avg_dir:
            avg_dir = self.compute_average_dir()
            error_map = np.arccos(
                np.clip(np.abs(np.sum(avg_dir * peak_img, axis=1)), 0.0, 1.0))
        else:
            error_map = np.zeros((len(peak_img)), dtype=float)
            for i in range(self.nb_sphere_vts):
                count_i = self.todi[:, i]
                error_i = np.dot(peak_img, self.sphere.vertices[i])
                mask = np.isfinite(error_i)
                arccos_i = np.arccos(np.clip(np.abs(error_i[mask]), 0.0, 1.0))
                error_map[mask] += count_i[mask] * arccos_i

            if normalize_count:
                tdi = self.get_tdi().astype(float)
                tdi_zero = tdi < MINIMUM_TODI_EPSILON
                error_map[tdi_zero] = 0.0
                error_map[~tdi_zero] /= tdi[~tdi_zero]

        if deg:
            error_map *= 180.0 / np.pi

        return error_map

    def compute_average_dir(self):
        """Voxel-wise average of TODI orientations.

        Average all orientation of each voxel, of the TODI map,
        into a single direction, warning for crossing.

        Returns
        -------
        avg_dir : numpy.ndarray (4D)
            Volume containing a single 3-vector (peak) per voxel.
        """
        avg_dir = np.zeros((len(self.todi), 3), dtype=float)

        sym_dir_index = self.nb_sphere_vts // 2
        for i in range(sym_dir_index):
            current_dir = self.sphere.vertices[i]
            count_dir = (self.todi[:, i] + self.todi[:, i + sym_dir_index])
            avg_dir += np.outer(count_dir, current_dir)

        avg_dir = todi_u.normalize_vectors(avg_dir)
        return avg_dir

    def __enter__(self):
        # Necessary for a 'with' statement to scrap a todi_object after
        # the scope of operation in the script scil_priors_from_streamlines.py
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # Necessary for a 'with' statement to scrap a todi_object after
        # the scope of operation in the script scil_priors_from_streamlines.py
        pass
