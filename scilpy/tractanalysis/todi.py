# -*- coding: utf-8 -*-
from copy import deepcopy
import logging

from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import scilpy.tractanalysis.todi_util as todi_u

MINIMUM_TODI_EPSILON = 1e-8
GAUSSIAN_TRUNCATE = 2.0


class TrackOrientationDensityImaging(object):
    def __init__(self, img_shape, sphere_type='repulsion724'):
        """Build the object

        At each voxel an histogram distribution of
        the local streamlines orientations (TODI) is computed.

        Parameters
        ----------
        img_shape : tuple, list, array
            Dimensions of the reference image
        sphere_type : str
            The distribution of orientation is discretize on that sphere
                [repulsion724]
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

    def compute_todi(self, streamlines, length_weights=True):
        """Compute the TODI map,
        :param streamlines: list or Array Sequence
        :param length_weights: bool, use length of each segment as weight
        """
        # Streamlines vertices in "VOXEL_SPACE" within "img_shape" range
        pts_pos, pts_dir, pts_norm = \
            todi_u.streamlines_to_pts_dir_norm(streamlines)

        if not length_weights:
            pts_norm = None

        sph_ids = todi_u.get_dir_to_sphere_id(pts_dir, self.sphere.vertices)

        # Get voxel indices for each point
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

        todi_bin_1d = np.bincount(
            np.ravel_multi_index(np.stack((pts_vox, sph_ids)), todi_bin_shape),
            weights=pts_norm, minlength=todi_bin_len)

        # Bincount of sphere id for each voxel
        self.todi = todi_bin_1d.reshape(todi_bin_shape)

    def get_todi(self):
        return self.todi

    def get_tdi(self):
        return np.sum(self.todi, axis=-1)

    def get_todi_shape(self):
        return self.todi_shape

    def get_mask(self):
        return self.mask

    def mask_todi(self, mask):
        """Mask the TODI map without having to reshape the whole volume all
        at once (big in memory)
        :param mask: numpy.ndarray, binary mask
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
        """Smooth the orientations / distribution on the sphere.
        Important for priors construction of BST
        :param order: int, blurring factor (based on the dot product)
        """
        assert order >= 1
        todi_sum = np.sum(self.todi, axis=-1, keepdims=True)
        sphere_dot = np.dot(self.sphere.vertices, self.sphere.vertices.T)
        sphere_psf = np.abs(sphere_dot) ** order
        self.todi = np.dot(self.todi, sphere_psf)
        self.todi *= todi_sum / np.sum(self.todi, axis=-1, keepdims=True)

    def smooth_todi_spatial(self, sigma=0.5):
        """Blur the TODI map using neighborhood information.
        Important for priors construction of BST. (RAM-friendly version)
        :param sigma: float, blurring factor
        """
        # This operation changes the mask as well as the TODI
        mask_3d = self.reshape_to_3d(self.mask).astype(np.float)
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
            self.todi = np.delete(self.todi, range(0, chunk_size), axis=1)
            chunk_count -= 1

        self.mask = new_mask
        self.todi = new_todi

    def normalize_todi_per_voxel(self, p_norm=2):
        """Normalize TODI with order 'p_norm'
        :param p_norm: int, norm type (default L2)
        :return numpy.ndarray, TODI (SF) masked array
        """
        self.todi = todi_u.p_normalize_vectors(self.todi, p_norm)
        return self.todi

    def get_sh(self, sh_basis, sh_order):
        """Compute the SH representation of the TODI
        :return numpy.ndarray, TODI (SH) masked array
        """
        return sf_to_sh(self.todi, self.sphere, sh_order, sh_basis, 0.006)

    def reshape_to_3d(self, img_voxelly_masked):
        """Reshape a unravel binary mask to its original shape
        :param img_voxelly_masked: numpy.ndarray, either in 1/2/3D that will be
            reshaped to 3D (input data shape) accordingly.
            (Necessary for future masking operation)
        :return img_voxelly_masked, numpy.ndarray (3D)
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
        """Compute the distance between "gold standard" peaks and a TODI map
            in radian or degree
        :param peak_img: numpy.ndarray (4D) contains peaks as written by most
            of our scripts
        :param normalize_count: bool, Normalize/weight the error map by the
            density map
        :param deg: bool, Error map will be return as degree instead of radian
        :param with_avg_dir: Average all orientation of a voxel of the TODI map
            into a single direction (warning for crossing)
        :return error_map, numpy.ndarray map of the cummulative radian error
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
            error_map = np.zeros((len(peak_img)), dtype=np.float)
            for i in range(self.nb_sphere_vts):
                count_i = self.todi[:, i]
                error_i = np.dot(peak_img, self.sphere.vertices[i])
                mask = np.isfinite(error_i)
                arccos_i = np.arccos(np.clip(np.abs(error_i[mask]), 0.0, 1.0))
                error_map[mask] += count_i[mask] * arccos_i

            if normalize_count:
                tdi = self.get_tdi().astype(np.float)
                tdi_zero = tdi < MINIMUM_TODI_EPSILON
                error_map[tdi_zero] = 0.0
                error_map[~tdi_zero] /= tdi[~tdi_zero]

        if deg:
            error_map *= 180.0 / np.pi

        return error_map

    def compute_average_dir(self):
        """Average all orientation (voxel-wise) of the TODI map
            into a single direction (warning for crossing)
        :return avg_dir, numpy.ndarray with single 3D vector per voxel
        """
        avg_dir = np.zeros((len(self.todi), 3), dtype=np.float)

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
