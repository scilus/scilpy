# -*- coding: utf-8 -*-

import nibabel as nib
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
                 original_axcodes=None):
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
        return StatefulImage(source.dataobj, source.affine,
                             header=source.header,
                             original_affine=reference._original_affine,
                             original_dimensions=reference._original_dimensions,
                             original_voxel_sizes=reference._original_voxel_sizes,
                             original_axcodes=reference._original_axcodes)

    @staticmethod
    def convert_to_simg(img):
        """
        Initialize a StatefulImage from an existing Nifti1Image.

        This constructor allows creating a StatefulImage directly from a
        Nifti1Image, preserving its original orientation information.

        Parameters
        ----------
        img : nib.Nifti1Image
            The Nifti1Image to initialize from.
        """
        return StatefulImage(img.dataobj, img.affine, header=img.header,
                             original_affine=img.affine.copy(),
                             original_dimensions=img.header.get_data_shape(),
                             original_voxel_sizes=img.header.get_zooms(),
                             original_axcodes=nib.orientations.aff2axcodes(
                                 img.affine))

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
        validate_voxel_order(target_axcodes)

        current_axcodes = nib.orientations.aff2axcodes(self.affine)
        if current_axcodes == tuple(target_axcodes):
            return

        # Check unique are only valid axis codes
        valid_codes = {'L', 'R', 'A', 'P', 'S', 'I'}
        for code in target_axcodes:
            if code not in valid_codes:
                raise ValueError(f"Invalid axis code '{code}' in target.")

        # Check L/R, A/P, S/I pairs are not both present
        pairs = [('L', 'R'), ('A', 'P'), ('S', 'I')]
        for pair in pairs:
            if pair[0] in target_axcodes and pair[1] in target_axcodes:
                raise ValueError(f"Conflicting axis codes '{pair[0]}' and "
                                 f"'{pair[1]}' in target.")

        # Check no repeated axis codes (LL, RR, etc.)
        if len(set(target_axcodes)) != 3:
            raise ValueError("Target axis codes must be unique.")

        start_ornt = nib.orientations.axcodes2ornt(current_axcodes)
        target_ornt = nib.orientations.axcodes2ornt(target_axcodes)
        transform = nib.orientations.ornt_transform(start_ornt, target_ornt)

        reoriented_img = self.as_reoriented(transform)
        self.__init__(reoriented_img.dataobj, reoriented_img.affine,
                      reoriented_img.header,
                      original_affine=self._original_affine,
                      original_dimensions=self._original_dimensions,
                      original_voxel_sizes=self._original_voxel_sizes,
                      original_axcodes=self._original_axcodes)

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
        self.reorient(voxel_order)

    @property
    def axcodes(self):
        """Get the axis codes for the current image orientation."""
        return nib.orientations.aff2axcodes(self.affine)

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
