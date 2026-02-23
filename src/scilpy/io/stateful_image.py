# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np

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
                 gradients_original_order=True):
        """
        Initialize a StatefulImage object.

        Extends the Nifti1Image constructor to store original orientation info.
        """
        super().__init__(dataobj, affine, header, extra, file_map)

        # Store original image information
        self._original_affine = original_affine
        self._original_dimensions = original_dimensions
        self._original_voxel_sizes = original_voxel_sizes
        self._original_axcodes = original_axcodes

        # Store gradient information
        self._bvals = None
        self._bvecs = None
        if bvals is not None and bvecs is not None:
            self.attach_gradients(bvals, bvecs, gradients_original_order)

    @classmethod
    def load(cls, filename, to_orientation="RAS"):
        """
        Load a NIfTI image, store its original orientation, and reorient it.

        Parameters
        ----------
        filename : str
            Path to the NIfTI file.
        to_orientation : str or tuple, optional
            The target orientation for the in-memory data. Default is "RAS".

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

        return cls(reoriented_img.dataobj, reoriented_img.affine,
                   reoriented_img.header, original_affine=original_affine,
                   original_dimensions=original_dims,
                   original_voxel_sizes=original_voxel_sizes,
                   original_axcodes=original_axcodes)

    def save(self, filename):
        """
        Save the image to a file, reverting to its original orientation.

        Parameters
        ----------
        filename : str
            Path to save the NIfTI file.
        """
        if self._original_axcodes is None:
            raise ValueError(
                "Unknown original orientation. Ensure the image was loaded"
                "with StatefulImage.load() or that original_axcodes was"
                "provided when creating the StatefulImage instance.")

        self.reorient_to_original()
        nib.save(self, filename)

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
        if reference.bvals is not None and reference.bvecs is not None:
            if source.ndim >= 4 and len(reference.bvals) == source.shape[3]:
                bvals = reference.bvals
                bvecs = reference.bvecs

                # If reference orientation != source orientation, reorient bvecs
                ref_axcodes = reference.axcodes
                source_axcodes_3d = nib.orientations.aff2axcodes(source.affine)
                
                if ref_axcodes[:3] != source_axcodes_3d:
                    # Strip 'T' etc. for nibabel
                    ref_axcodes_3d = ref_axcodes[:3]

                    # Use a temporary StatefulImage logic to reorient bvecs
                    start_ornt = nib.orientations.axcodes2ornt(ref_axcodes_3d)
                    target_ornt = nib.orientations.axcodes2ornt(source_axcodes_3d)
                    transform = nib.orientations.ornt_transform(start_ornt, target_ornt)
                    axis_permutation = transform[:, 0].astype(int)
                    axis_flips = transform[:, 1]
                    bvecs = bvecs[:, axis_permutation] * axis_flips

        return StatefulImage(source.dataobj, source.affine,
                             header=source.header,
                             original_affine=reference._original_affine,
                             original_dimensions=reference._original_dimensions,
                             original_voxel_sizes=reference._original_voxel_sizes,
                             original_axcodes=reference._original_axcodes,
                             bvals=bvals, bvecs=bvecs,
                             gradients_original_order=False)

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

    @property
    def bvals(self):
        """Get the current b-values."""
        return self._bvals

    @property
    def bvecs(self):
        """Get the current (reoriented) b-vectors."""
        return self._bvecs

    def attach_gradients(self, bvals, bvecs, original_order=True):
        """
        Attach b-values and b-vectors to the image.

        Parameters
        ----------
        bvals : array-like
            B-values.
        bvecs : array-like
            B-vectors.
        original_order : bool, optional
            If True, assumes b-vectors are in the original voxel order.
            If False, assumes b-vectors match the current in-memory orientation.
            Default is True.
        """
        self._bvals = np.asanyarray(bvals)
        self._bvecs = np.asanyarray(bvecs)

        # Validate shapes
        if self._bvals.ndim != 1:
            raise ValueError("bvals must be a 1D array.")
        if self._bvecs.ndim != 2 or self._bvecs.shape[1] != 3:
            raise ValueError("bvecs must be an (N, 3) array.")
        if len(self._bvals) != len(self._bvecs):
            raise ValueError("bvals and bvecs must have the same length.")

        # Validate against image data
        if len(self._bvals) != self.shape[3]:
            raise ValueError(f"Number of gradients ({len(self._bvals)}) does "
                             f"not match number of volumes ({self.shape[3]}).")

        # If current orientation is not original, and we assume original, reorient
        if original_order and self.axcodes != self._original_axcodes:
            self._reorient_gradients(self._original_axcodes, self.axcodes)

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
        if self._bvals is None or self._bvecs is None:
            raise ValueError("No gradients attached to this StatefulImage.")

        # Reorient back to original for saving
        bvecs_to_save = self._bvecs
        if self.axcodes != self._original_axcodes:
            # We don't want to modify self._bvecs in-place here if we just
            # want to save. But simg.save() reorients the whole image back!
            # So if we follow that pattern, we should probably reorient
            # back, save, and then (if needed) reorient back to current.
            # However, simg.save() calls reorient_to_original() which DOES
            # modify in-place.
            self.reorient_to_original()
            bvecs_to_save = self._bvecs

        np.savetxt(bvec_path, bvecs_to_save.T, fmt='%.8f')
        np.savetxt(bval_path, self._bvals[None, :], fmt='%.3f')

    def _reorient_gradients(self, start_axcodes, target_axcodes):
        """
        Internal helper to reorient b-vectors.

        Parameters
        ----------
        start_axcodes : tuple
            Starting axis codes.
        target_axcodes : tuple
            Target axis codes.
        """
        if self._bvecs is None:
            return

        # Strip 'T' if present
        start_axcodes_3d = [c for c in start_axcodes if c != 'T']
        target_axcodes_3d = [c for c in target_axcodes if c != 'T']

        start_ornt = nib.orientations.axcodes2ornt(start_axcodes_3d)
        target_ornt = nib.orientations.axcodes2ornt(target_axcodes_3d)
        transform = nib.orientations.ornt_transform(start_ornt, target_ornt)

        axis_permutation = transform[:, 0].astype(int)
        axis_flips = transform[:, 1]

        # Apply permutation and flips
        # bvecs is (N, 3). We permute columns and multiply by flips.
        self._bvecs = self._bvecs[:, axis_permutation] * axis_flips

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
                "Original axis codes are not set cannot reorient to original"
                "orientation.")
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
        target_axcodes = list(target_axcodes)
        if len(target_axcodes) < len(self.shape):
            extra_codes = ['T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            for i in range(len(target_axcodes), len(self.shape)):
                target_axcodes.append(extra_codes[i-3])
        elif len(target_axcodes) > len(self.shape):
            target_axcodes = target_axcodes[:len(self.shape)]
        target_axcodes = tuple(target_axcodes)

        validate_voxel_order(target_axcodes, dimensions=len(self.shape))

        current_axcodes = self.axcodes
        if current_axcodes == tuple(target_axcodes):
            return

        # Nibabel only handles 3D orientations. If 4D, we assume the 4th
        # dimension is time/gradients and doesn't need reorientation.
        target_axcodes_3d = [c for c in target_axcodes if c != 'T']
        current_axcodes_3d = [c for c in current_axcodes if c != 'T']

        start_ornt = nib.orientations.axcodes2ornt(current_axcodes_3d)
        target_ornt = nib.orientations.axcodes2ornt(target_axcodes_3d)
        transform = nib.orientations.ornt_transform(start_ornt, target_ornt)

        reoriented_img = self.as_reoriented(transform)

        # Reorient gradients before re-initializing
        if self._bvecs is not None:
            self._reorient_gradients(current_axcodes, target_axcodes)

        # Pass current reoriented gradients to __init__
        self.__init__(reoriented_img.dataobj, reoriented_img.affine,
                      reoriented_img.header,
                      original_affine=self._original_affine,
                      original_dimensions=self._original_dimensions,
                      original_voxel_sizes=self._original_voxel_sizes,
                      original_axcodes=self._original_axcodes,
                      bvals=self._bvals, bvecs=self._bvecs,
                      gradients_original_order=False)

        # Mark that these gradients are already in target orientation
        # wait, __init__ will call attach_gradients(bvals, bvecs, original_order=True)
        # by default. I need to change how __init__ calls it if it's from here.

        # I'll update __init__ to accept original_order flag.

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
            Reference object from which orientation information can be obtained.
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
                codes.append(extra_codes[i-3])
        return tuple(codes)

    @property
    def original_axcodes(self):
        """Get the axis codes for the original image orientation."""
        return self._original_axcodes

    def __str__(self):
        """Return a string representation of the image, including orientation."""
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
