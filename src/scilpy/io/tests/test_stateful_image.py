# -*- coding: utf-8 -*-

import os
import pytest
import nibabel as nib
import numpy as np

from scilpy.io.stateful_image import StatefulImage


@pytest.fixture
def dummy_nifti_file(tmp_path):
    """
    Create a dummy NIfTI file for testing.
    """
    shape = (10, 10, 10)
    affine = np.eye(4)
    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine)
    
    # Save the image to a temporary file
    file_path = os.path.join(tmp_path, "test.nii.gz")
    nib.save(img, file_path)
    
    return file_path, affine, shape


def test_load_and_reorient(dummy_nifti_file):
    """
    Test loading a NIfTI file and reorienting it to RAS.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path, to_orientation='RAS')

    assert isinstance(img, StatefulImage)
    assert img.axcodes == ('R', 'A', 'S')
    assert img.original_axcodes == ('R', 'A', 'S')


def test_save_to_original_orientation(dummy_nifti_file, tmp_path):
    """
    Test that saving the image reverts it to its original orientation.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path, to_orientation='LPS')

    # Save the image
    output_path = os.path.join(tmp_path, "output.nii.gz")
    img.save(output_path)

    # Load the saved image and check its orientation
    saved_img = nib.load(output_path)
    assert nib.orientations.aff2axcodes(saved_img.affine) == ('R', 'A', 'S')


def test_reorient_to_original(dummy_nifti_file):
    """
    Test reorienting the image back to its original orientation.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path, to_orientation='LPS')
    img.reorient_to_original()
    assert img.axcodes == ('R', 'A', 'S')


def test_reorient_invalid_codes(dummy_nifti_file):
    """
    Test that reorienting with invalid codes raises a ValueError.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path)
    with pytest.raises(ValueError):
        img.reorient(('X', 'Y', 'Z'))


def test_reorient_conflicting_codes(dummy_nifti_file):
    """
    Test that reorienting with conflicting codes raises a ValueError.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path)
    with pytest.raises(ValueError):
        img.reorient(('L', 'R', 'S'))


def test_reorient_non_unique_codes(dummy_nifti_file):
    """
    Test that reorienting with non-unique codes raises a ValueError.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path)
    with pytest.raises(ValueError):
        img.reorient(('L', 'L', 'S'))


def test_to_ras_lps(dummy_nifti_file):
    """
    Test the to_ras() and to_lps() convenience methods.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path)
    
    img.to_lps()
    assert img.axcodes == ('L', 'P', 'S')

    img.to_ras()
    assert img.axcodes == ('R', 'A', 'S')


def test_to_reference(dummy_nifti_file, tmp_path):
    """
    Test reorienting to match a reference image.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path)

    # Create a reference image with a different orientation
    ref_affine = np.diag([-1, -1, 1, 1])
    ref_img = nib.Nifti1Image(np.zeros((10, 10, 10)), ref_affine)
    
    img.to_reference(ref_img)
    assert img.axcodes == ('L', 'P', 'S')


def test_to_reference_stateful_image(dummy_nifti_file):
    """
    Test that to_reference raises a TypeError with a StatefulImage.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path)
    ref_img = StatefulImage.load(file_path)

    with pytest.raises(TypeError):
        img.to_reference(ref_img)


def test_axcodes_properties(dummy_nifti_file):
    """
    Test the axcodes and original_axcodes properties.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path, to_orientation='LPS')
    assert img.axcodes == ('L', 'P', 'S')
    assert img.original_axcodes == ('R', 'A', 'S')


def test_str_representation(dummy_nifti_file):
    """
    Test the string representation of the StatefulImage.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path, to_orientation='LPS')
    s = str(img)
    assert "Original axis codes:    ('R', 'A', 'S')" in s
    assert "Current axis codes:     ('L', 'P', 'S')" in s
    assert "Reoriented from original: True" in s


def test_load_no_reorientation(dummy_nifti_file):
    """
    Test that loading without reorientation works as expected.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path, to_orientation=None)
    assert img.axcodes == ('R', 'A', 'S')
    assert img.original_axcodes == ('R', 'A', 'S')


def test_reorient_no_op(dummy_nifti_file):
    """
    Test that reorienting to the same orientation is a no-op.
    """
    file_path, _, _ = dummy_nifti_file
    img = StatefulImage.load(file_path)
    img.reorient(('R', 'A', 'S'))
    assert img.axcodes == ('R', 'A', 'S')


def test_direct_instantiation(dummy_nifti_file):
    """
    Test direct instantiation of StatefulImage.
    """
    file_path, affine, shape = dummy_nifti_file
    nii = nib.load(file_path)
    img = StatefulImage(nii.dataobj, nii.affine, nii.header)

    assert img.original_axcodes is None
    assert img.axcodes == ('R', 'A', 'S')

    # Test that save fails without original orientation information
    with pytest.raises(ValueError):
        img.save("test.nii.gz")

    # Test that reorient_to_original fails
    with pytest.raises(ValueError):
        img.reorient_to_original()