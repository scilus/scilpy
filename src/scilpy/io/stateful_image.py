# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
from scipy.linalg import polar

from dipy.io.gradients import read_bvals_bvecs
from dipy.io.utils import get_reference_info
from scilpy.utils.orientation import validate_voxel_order


class StatefulImage(nib.Nifti1Image):
    """
    A class that extends nib.Nifti1Image to manage image orientation state.

    This class ensures that image data loaded into memory is always in a
    consistent orientation (RAS by default), while preserving the original
    on-disk orientation information. When saving, the image is automatically
    reverted to its original orientation, ensuring non-destructive operations.
    """

    def __init__(self, dataobj, affine, header=None, extra=None,
                 file_map=None, original_affine=None,
                 original_dimensions=None, original_voxel_sizes=None,
                 original_axcodes=None, bvals=None, bvecs=None,
                 gradients_original_order=True,
                 sh_basis='descoteaux07', is_legacy=True,
                 is_orientation=False, is_world_space=True):
        """
        Initialize a StatefulImage object.

        Extends the Nifti1Image constructor to store original orientation info.
        """
        super().__init__(dataobj, affine, header, extra, file_map)

        # Store original image information
        self._original_affine = original_affine \
            if original_affine is not None else affine.copy()
        self._original_dimensions = original_dimensions \
            if original_dimensions is not None else self.header.get_data_shape()
        self._original_voxel_sizes = original_voxel_sizes \
            if original_voxel_sizes is not None else self.header.get_zooms()
        self._original_axcodes = original_axcodes \
            if original_axcodes is not None else \
            nib.orientations.aff2axcodes(affine)

        # Directional information
        self._sh_basis = sh_basis
        self._is_legacy = is_legacy
        self._is_orientation = is_orientation
        self._is_world_space = is_world_space

        # Store gradient information
        self._bvals = None
        self._world_bvecs = None
        if bvals is not None and bvecs is not None:
            self.attach_gradients(bvals, bvecs, gradients_original_order)

    @staticmethod
    def _get_rotation_matrix(affine):
        """
        Extract the pure rotation component from a 4x4 affine matrix.
        """
        # Extract 3x3 part
        A = affine[:3, :3]
        # Polar decomposition: A = P * R
        # R is the closest orthogonal matrix to A.
        # We want the orthogonal part that matches the image's orientation.
        R, P = polar(A)
        return R

    @classmethod
    def load(cls, filename, to_orientation="RAS",
             is_orientation=False, is_world_space=True,
             sh_basis='descoteaux07', is_legacy=True):
        """
        Load a NIfTI image, store its original orientation, and reorient it.

        Parameters
        ----------
        filename : str
            Path to the NIfTI file.
        to_orientation : str or tuple, optional
            The target orientation for the in-memory data. Default is "RAS".
        is_orientation : bool, optional
            Whether the image contains directional data (SH, Peaks, SF).
            Default is False.
        is_world_space : bool, optional
            Whether the directional data is already in world space.
            Only used if is_orientation is True. Default is True.

        Returns
        -------
        StatefulImage
            An instance of StatefulImage with data in the target orientation.
        """
        img = nib.load(filename)

        original_affine = img.affine.copy()
        original_axcodes = nib.orientations.aff2axcodes(img.affine)
        original_dims = img.header.get_data_shape()
        original_voxel_sizes = img.header.get_zooms()

        if to_orientation:
            validate_voxel_order(to_orientation)
            start_ornt = nib.orientations.io_orientation(img.affine)
            target_ornt = nib.orientations.axcodes2ornt(to_orientation)
            transform = nib.orientations.ornt_transform(start_ornt,
                                                        target_ornt)
            reoriented_img = img.as_reoriented(transform)
        else:
            reoriented_img = img

        simg = cls(reoriented_img.dataobj, reoriented_img.affine,
                   reoriented_img.header, original_affine=original_affine,
                   original_dimensions=original_dims,
                   original_voxel_sizes=original_voxel_sizes,
                   original_axcodes=original_axcodes,
                   sh_basis=sh_basis, is_legacy=is_legacy,
                   is_orientation=is_orientation,
                   is_world_space=is_world_space)

        if is_orientation and not is_world_space:
            # Move from original voxel space to world space
            # Note: We use original_affine because the data was loaded
            # in that space.
            data = simg.get_fdata(dtype=np.float32)
            R = simg._get_rotation_matrix(original_affine)
            rotated_data = simg._rotate_direction_data(data, R,
                                                       sh_basis=sh_basis,
                                                       is_legacy=is_legacy)
            simg = cls.from_data(rotated_data, simg)
            simg._is_world_space = True

        return simg

    def to_voxel_direction(self, data=None, sh_basis=None,
                           is_legacy=None, nbr_processes=None):
        """
        Transform directional data from world space to current voxel space.

        Parameters
        ----------
        data : np.ndarray, optional
            The directional data to transform. If None, uses the image data
            and updates it in-place.
        sh_basis : str, optional
            The SH basis of the directional data. Defaults to self.sh_basis.
        is_legacy : bool, optional
            Whether the SH basis is legacy. Defaults to self.is_legacy.
        nbr_processes : int, optional
            Number of processes to use for rotation.

        Returns
        -------
        np.ndarray
            The transformed directional data in voxel space.
        """
        if sh_basis is None:
            sh_basis = self.sh_basis
        if is_legacy is None:
            is_legacy = self.is_legacy

        if data is None:
            if not self.is_orientation:
                raise ValueError("Image is not marked as directional.")
            if not self.is_world_space:
                return self.get_fdata(dtype=np.float32)

            data = self.get_fdata(dtype=np.float32)
            R = self._get_rotation_matrix(self.affine).T
            rotated_data = self._rotate_direction_data(
                data, R, sh_basis=sh_basis, is_legacy=is_legacy, nbr_processes=nbr_processes)
            self._dataobj = rotated_data
            self._is_world_space = False
            return rotated_data

        # R_world_to_voxel = R_voxel_to_world.T
        R = self._get_rotation_matrix(self.affine).T
        return self._rotate_direction_data(data, R, sh_basis=sh_basis,
                                           is_legacy=is_legacy,
                                           nbr_processes=nbr_processes)

    def to_world_direction(self, data=None, sh_basis=None,
                           is_legacy=None, nbr_processes=None):
        """
        Transform directional data from voxel space to world space.

        Parameters
        ----------
        data : np.ndarray, optional
            The directional data to transform. If None, uses the image data
            and updates it in-place.
        sh_basis : str, optional
            The SH basis of the directional data. Defaults to self.sh_basis.
        is_legacy : bool, optional
            Whether the SH basis is legacy. Defaults to self.is_legacy.
        nbr_processes : int, optional
            Number of processes to use for rotation.

        Returns
        -------
        np.ndarray
            The transformed directional data in world space.
        """
        if sh_basis is None:
            sh_basis = self.sh_basis
        if is_legacy is None:
            is_legacy = self.is_legacy

        if data is None:
            if not self.is_orientation:
                raise ValueError("Image is not marked as directional.")
            if self.is_world_space:
                return self.get_fdata(dtype=np.float32)

            data = self.get_fdata(dtype=np.float32)
            R = self._get_rotation_matrix(self.affine)
            rotated_data = self._rotate_direction_data(
                data, R, sh_basis=sh_basis, is_legacy=is_legacy, nbr_processes=nbr_processes)
            self._dataobj = rotated_data
            self._is_world_space = True
            return rotated_data

        R = self._get_rotation_matrix(self.affine)
        return self._rotate_direction_data(data, R, sh_basis=sh_basis,
                                           is_legacy=is_legacy,
                                           nbr_processes=nbr_processes)

    def _rotate_direction_data(self, data, R, sh_basis='descoteaux07',
                               is_legacy=True, nbr_processes=None):
        """
        Internal helper to rotate SH or Peaks data.
        """
        from scilpy.reconst.utils import (get_sh_order_and_fullness,
                                          is_data_peaks)

        original_shape = data.shape
        if len(original_shape) == 5 and original_shape[-1] == 7:
            # Bingham-like data: [amp, mu1_x, mu1_y, mu1_z, mu2_x, mu2_y, mu2_z]
            # We rotate mu1 and mu2
            bingham_data = data.copy()
            mu1 = bingham_data[..., 1:4].reshape(-1, 3)
            mu2 = bingham_data[..., 4:7].reshape(-1, 3)
            rotated_mu1 = np.dot(mu1, R.T)
            rotated_mu2 = np.dot(mu2, R.T)
            bingham_data[..., 1:4] = rotated_mu1.reshape(
                original_shape[:4] + (3,))
            bingham_data[..., 4:7] = rotated_mu2.reshape(
                original_shape[:4] + (3,))
            return bingham_data

        # Handle 5D data
        if len(original_shape) == 5:
            # We treat each "lobe" independently for rotation if it's not SH
            data = data.reshape(original_shape[0:3] + (-1,))

        last_dim = data.shape[-1]
        is_sh = not is_data_peaks(data)
        if is_sh:
            from scilpy.reconst.sh import rotate_sh
            # SH data can be 4D (XxYxZxN)
            order, full = get_sh_order_and_fullness(last_dim)
            return rotate_sh(data, R, basis_type=sh_basis,
                             full_basis=full, is_legacy=is_legacy,
                             nbr_processes=nbr_processes)
        elif last_dim % 3 == 0:
            # Assume Peaks (N*3)
            # Reshape to (..., N, 3), rotate, and reshape back
            reshaped_data = data.reshape(-1, 3)
            rotated_data = np.dot(reshaped_data, R.T)
            return rotated_data.reshape(original_shape)
        else:
            raise ValueError(
                f"Could not identify directional data type for "
                f"shape {original_shape}. Not SH (wrong #coeffs) and "
                f"not Peaks (not a multiple of 3).")

    def save(self, filename):
        """
        Save the image to a file, reverting to its original orientation.

        Parameters
        ----------
        filename : str
            Path to save the NIfTI file.
        """
        current_axcodes = self.axcodes[:3]
        target_axcodes = self._original_axcodes[:3]

        # 1. Handle directional data
        if self.is_orientation:
            # Bring to current voxel space
            data = self.to_voxel_direction()

            if current_axcodes != target_axcodes:
                # Need to rotate from current voxel space to target voxel space
                R_curr = self._get_rotation_matrix(self.affine)
                R_target = self._get_rotation_matrix(self._original_affine)
                # target = R_target.T * world = R_target.T * (R_curr * current)
                R = np.dot(R_target.T, R_curr)
                data = self._rotate_direction_data(data, R,
                                                   sh_basis=self.sh_basis,
                                                   is_legacy=self.is_legacy)

            # Create a temporary image for reorientation
            temp_img = nib.Nifti1Image(data, self.affine, self.header)
        else:
            temp_img = self

        # 2. Reorient voxels to original orientation
        if current_axcodes != target_axcodes:
            start_ornt = nib.orientations.axcodes2ornt(current_axcodes)
            target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
            transform = nib.orientations.ornt_transform(start_ornt,
                                                        target_ornt)
            # Use Nifti1Image.as_reoriented to get a temporary object
            # in the original orientation for saving.
            final_img = temp_img.as_reoriented(transform)
        else:
            final_img = temp_img

        nib.save(final_img, filename)

    @staticmethod
    def create_from(source, reference):
        """
        Create a new StatefulImage from a source image, preserving the original
        orientation information from a reference StatefulImage.

        Parameters
        ----------
        source : nib.Nifti1Image
            The image data to use for the new StatefulImage.
        reference : StatefulImage
            The reference image from which to copy original orientation
            information.

        Returns
        -------
        StatefulImage
            A new StatefulImage with the source image's data and the reference
            image's original orientation information.
        """
        bvals = None
        bvecs = None
        if reference.bvals is not None and reference.world_bvecs is not None:
            if source.ndim == 4 and len(reference.bvals) == source.shape[3]:
                bvals = reference.bvals
                # Transform world-space bvecs to source voxel space
                R_source = reference._get_rotation_matrix(source.affine)
                bvecs = np.dot(reference.world_bvecs, R_source)

                if StatefulImage.needs_fsl_flip(source.affine):
                    bvecs[:, 0] *= -1
        orig_dims = reference._original_dimensions
        orig_vox = reference._original_voxel_sizes
        return StatefulImage(source.dataobj, source.affine,
                             header=source.header,
                             original_affine=reference._original_affine,
                             original_dimensions=orig_dims,
                             original_voxel_sizes=orig_vox,
                             original_axcodes=reference._original_axcodes,
                             bvals=bvals, bvecs=bvecs,
                             gradients_original_order=False,
                             sh_basis=reference.sh_basis,
                             is_legacy=reference.is_legacy,
                             is_orientation=reference.is_orientation,
                             is_world_space=reference.is_world_space)

    @staticmethod
    def from_data(data, reference, is_orientation=None):
        """
        Create a new StatefulImage from a numpy array, preserving the original
        orientation information from a reference StatefulImage.

        Parameters
        ----------
        data : numpy.ndarray
            The image data to use for the new StatefulImage.
        reference : StatefulImage
            The reference image from which to copy original orientation
            information.
        is_orientation : bool, optional
            Whether the new image contains directional data.
            If None, uses reference.is_orientation.

        Returns
        -------
        StatefulImage
            A new StatefulImage with the data and the reference
            image's original orientation information.
        """
        new_img = nib.Nifti1Image(data, reference.affine, reference.header)
        simg = StatefulImage.create_from(new_img, reference)
        if is_orientation is not None:
            simg._is_orientation = is_orientation
        return simg

    @staticmethod
    def convert_to_simg(img, bvals=None, bvecs=None):
        """
        Initialize a StatefulImage from an existing Nifti1Image.

        This constructor allows creating a StatefulImage directly from a
        Nifti1Image, preserving its original orientation information.

        Parameters
        ----------
        img : nib.Nifti1Image
            The Nifti1Image to initialize from.
        bvals : array-like, optional
            B-values.
        bvecs : array-like, optional
            B-vectors.
        """
        original_axcodes = nib.orientations.aff2axcodes(img.affine)
        if len(img.shape) == 4:
            original_axcodes += ('T',)

        return StatefulImage(img.dataobj, img.affine, header=img.header,
                             original_affine=img.affine.copy(),
                             original_dimensions=img.header.get_data_shape(),
                             original_voxel_sizes=img.header.get_zooms(),
                             original_axcodes=original_axcodes,
                             bvals=bvals, bvecs=bvecs)

    @staticmethod
    def needs_fsl_flip(affine):
        """
        According to BIDS/MRtrix convention, if the determinant of the
        3x3 rotation/scaling part of the affine is positive (neurological),
        the x-component of the FSL-format bvecs must be flipped.
        """
        return np.linalg.det(affine[:3, :3]) > 0

    @property
    def _needs_fsl_flip(self):
        return StatefulImage.needs_fsl_flip(self.affine)

    @property
    def sh_basis(self):
        """Get the SH basis."""
        return self._sh_basis

    @property
    def is_legacy(self):
        """Get whether the SH basis is legacy."""
        return self._is_legacy

    @property
    def is_orientation(self):
        """Get whether the image contains directional data."""
        return self._is_orientation

    @property
    def is_world_space(self):
        """Get whether the directional data is in world space."""
        return self._is_world_space

    @property
    def bvals(self):
        """Get the current b-values."""
        return self._bvals

    @property
    def bvecs(self):
        """Get the current (reoriented) b-vectors in voxel space."""
        if self._world_bvecs is None:
            return None
        # Transform from world space to current voxel space
        R = self._get_rotation_matrix(self.affine)
        # v_voxel = v_world * R
        bvecs = np.dot(self._world_bvecs, R)

        if self._needs_fsl_flip:
            bvecs[:, 0] *= -1

        return bvecs

    @property
    def world_bvecs(self):
        """Get the current b-vectors in world space."""
        return self._world_bvecs

    def attach_gradients(self, bvals, bvecs, original_order=True):
        """
        Attach b-values and b-vectors to the image.
        Gradients are stored internally in world space.

        Parameters
        ----------
        bvals : array-like
            B-values.
        bvecs : array-like
            B-vectors.
        original_order : bool, optional
            If True, assumes b-vectors are in the original voxel order.
            If False, assumes b-vectors match current in-memory orientation.
            Default is True.
        """
        self._bvals = np.asanyarray(bvals)
        bvecs = np.asanyarray(bvecs).copy()

        # Validate shapes
        if self._bvals.ndim != 1:
            raise ValueError("bvals must be a 1D array.")
        if bvecs.ndim != 2 or bvecs.shape[1] != 3:
            raise ValueError("bvecs must be an (N, 3) array.")
        if len(self._bvals) != len(bvecs):
            raise ValueError("bvals and bvecs must have the same length.")

        # Validate against image data
        if len(self._bvals) != self.shape[3]:
            raise ValueError(f"Number of gradients ({len(self._bvals)}) does "
                             f"not match number of volumes ({self.shape[3]}).")

        if original_order:
            # Transform from original voxel space to world space
            ref_affine = self._original_affine \
                if self._original_affine is not None else self.affine
        else:
            # Transform from current voxel space to world space
            ref_affine = self.affine

        R = self._get_rotation_matrix(ref_affine)

        # Apply BIDS flip if needed
        if StatefulImage.needs_fsl_flip(ref_affine):
            bvecs[:, 0] *= -1

        self._world_bvecs = np.dot(bvecs, R.T)

        # Normalize
        norms = np.linalg.norm(self._world_bvecs, axis=1)
        self._world_bvecs[norms > 1e-6] /= norms[norms > 1e-6][:, None]

    def attach_world_gradients(self, bvals, world_bvecs):
        """
        Attach b-values and world-space b-vectors to the image.

        Parameters
        ----------
        bvals : array-like
            B-values.
        world_bvecs : array-like
            B-vectors in world space (RAS mm).
        """
        self._bvals = np.asanyarray(bvals)
        self._world_bvecs = np.asanyarray(world_bvecs).copy()

        # Validate shapes
        if self._bvals.ndim != 1:
            raise ValueError("bvals must be a 1D array.")
        if self._world_bvecs.ndim != 2 or self._world_bvecs.shape[1] != 3:
            raise ValueError("world_bvecs must be an (N, 3) array.")
        if len(self._bvals) != len(self._world_bvecs):
            raise ValueError(
                "bvals and world_bvecs must have the same length.")

        # Normalize
        norms = np.linalg.norm(self._world_bvecs, axis=1)
        self._world_bvecs[norms > 1e-6] /= norms[norms > 1e-6][:, None]

    def load_gradients(self, bval_path, bvec_path):
        """
        Load b-values and b-vectors from FSL-formatted files.

        Parameters
        ----------
        bval_path : str
            Path to the bvals file.
        bvec_path : str
            Path to the bvecs file.
        """
        bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
        self.attach_gradients(bvals, bvecs)

    def save_gradients(self, bval_path, bvec_path):
        """
        Save b-values and b-vectors to FSL-formatted files.
        Ensures b-vectors match the original voxel order.

        Parameters
        ----------
        bval_path : str
            Path to save the bvals file.
        bvec_path : str
            Path to save the bvecs file.
        """
        if self._bvals is None or self._world_bvecs is None:
            raise ValueError("No gradients attached to this StatefulImage.")

        # Transform from world space back to original voxel space
        ref_affine = self._original_affine \
            if self._original_affine is not None else self.affine
        R = self._get_rotation_matrix(ref_affine)
        # v_voxel = v_world * R
        bvecs_to_save = np.dot(self._world_bvecs, R)

        # According to BIDS/MRtrix convention, if the determinant of the
        # affine is positive (neurological), the x-component of the bvecs
        # must be flipped.
        if StatefulImage.needs_fsl_flip(ref_affine):
            bvecs_to_save[:, 0] *= -1

        np.savetxt(bvec_path, bvecs_to_save.T, fmt='%.8f')
        np.savetxt(bval_path, self._bvals[None, :], fmt='%.3f')

    def _reorient_gradients(self, start_axcodes, target_axcodes):
        """
        Internal helper to reorient b-vectors.
        Now that b-vectors are in world space, this does nothing.
        """
        pass

    def reorient_to_original(self):
        """
        Reorient the in-memory image to its original orientation.
        This method modifies the image in place. It does not return a new
        Nifti1Image instance.

        Raises
        ------
        ValueError
            If the original axis codes are not set.
        """
        if self._original_axcodes is None:
            raise ValueError(
                "Original axis codes are not set. Cannot reorient to original"
                " orientation.")
        self.reorient(self._original_axcodes)

    def reorient(self, target_axcodes):
        """
        Reorient the in-memory image to a target orientation.

        Parameters
        ----------
        target_axcodes : str or tuple
            The target orientation axis codes (e.g., "LPS", ("R", "A", "S")).
        """
        if target_axcodes is None:
            raise ValueError("Axis codes cannot be None.")

        # Ensure target_axcodes has the same number of dimensions as self.shape
        # by padding with unique placeholder codes if necessary.
        target_axcodes = tuple(target_axcodes[:3])

        validate_voxel_order(target_axcodes, dimensions=3)

        current_axcodes = self.axcodes[:3]
        if current_axcodes == target_axcodes:
            return

        start_ornt = nib.orientations.axcodes2ornt(current_axcodes)
        target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
        transform = nib.orientations.ornt_transform(start_ornt, target_ornt)

        # Use Nifti1Image.as_reoriented to get a temporary object
        # with the new orientation.
        reoriented_img = nib.Nifti1Image.as_reoriented(self, transform)

        # Update Nifti1Image attributes in-place
        self._dataobj = reoriented_img.dataobj
        self._affine = reoriented_img.affine
        self._header = reoriented_img.header

    def to_ras(self):
        """Convenience method to reorient in-memory data to RAS."""
        self.reorient(("R", "A", "S"))

    def to_lps(self):
        """Convenience method to reorient in-memory data to LPS."""
        self.reorient(("L", "P", "S"))

    def to_reference(self, obj):
        """
        Reorient the in-memory image to match the orientation of a reference
        object.

        Parameters
        ----------
        obj : object
            Reference object from which orientation information is obtained.
            Must not be an instance of ``StatefulImage``.

        Raises
        ------
        TypeError
            If ``obj`` is an instance of ``StatefulImage``.
        """

        if isinstance(obj, StatefulImage):
            raise TypeError('Reference object must not be a StatefulImage.')

        _, _, _, voxel_order = get_reference_info(obj)
        self.reorient(voxel_order[:3])

    @property
    def axcodes(self):
        """Get the axis codes for the current image orientation."""
        codes = list(nib.orientations.aff2axcodes(self.affine))
        if len(self.shape) > 3:
            extra_codes = ['T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            for i in range(3, len(self.shape)):
                codes.append(extra_codes[i - 3])
        return tuple(codes)

    @property
    def original_axcodes(self):
        """Get the axis codes for the original image orientation."""
        return self._original_axcodes

    @property
    def original_affine(self):
        """Get the original image affine."""
        return self._original_affine

    @property
    def original_header(self):
        """Get a header matching the original image orientation."""
        # Create a copy of the current header but with original info
        header = self.header.copy()
        header.set_sform(self._original_affine)
        header.set_qform(self._original_affine)
        if self._original_voxel_sizes is not None:
            header.set_zooms(self._original_voxel_sizes)
        if self._original_dimensions is not None:
            header.set_data_shape(self._original_dimensions)
        return header

    def __str__(self):
        """Return a string representation including orientation information."""
        base_str = super().__str__()
        current_axcodes = self.axcodes
        reoriented = current_axcodes != self._original_axcodes

        orientation_info = (
            f"Reorientation Information:\n"
            f"  Original axis codes:    {self._original_axcodes}\n"
            f"  Current axis codes:     {current_axcodes}\n"
            f"  Reoriented from original: {reoriented}"
        )

        return f"{base_str}\n{orientation_info}"
