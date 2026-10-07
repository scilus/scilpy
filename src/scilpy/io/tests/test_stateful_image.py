# -*- coding: utf-8 -*-

from contextlib import contextmanager
import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from scilpy.io.stateful_image import StatefulImage


@contextmanager
def create_dummy_nifti_file(filename="test.nii.gz", in_lps=False):
    """
    Create a dummy NIfTI file for testing in a temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (10, 10, 10)
        affine = np.eye(4) if not in_lps else np.diag([-1, -1, 1, 1])
        data = np.random.rand(*shape).astype(np.float32)
        img = nib.Nifti1Image(data, affine)

        file_path = os.path.join(tmpdir, filename)
        nib.save(img, file_path)

        yield file_path


def test_load_and_reorient():
    """
    Test loading a NIfTI file and reorienting it to RAS.
    """
    with create_dummy_nifti_file() as file_path:
        img = StatefulImage.load(file_path, to_orientation="RAS")

        assert isinstance(img, StatefulImage)
        assert img.axcodes == ("R", "A", "S")
        assert img.original_axcodes == ("R", "A", "S")


def test_save_to_original_orientation():
    """
    Test that saving the image reverts it to its original orientation.
    """
    with create_dummy_nifti_file() as file_path:
        img = StatefulImage.load(file_path, to_orientation="LPS")

        # Save the image
        tmp_dir = os.path.dirname(file_path)
        output_path = os.path.join(tmp_dir, "output.nii.gz")
        img.save(output_path)

        # Load the saved image and check its orientation
        saved_img = nib.load(output_path)
        assert nib.orientations.aff2axcodes(saved_img.affine) == ("R", "A", "S")


def test_reorient_to_original():
    """
    Test reorienting the image back to its original orientation.
    """
    with create_dummy_nifti_file() as file_path:
        img = StatefulImage.load(file_path, to_orientation="LPS")
        img.reorient_to_original()
        assert img.axcodes == ("R", "A", "S")


def test_to_ras_lps():
    """
    Test the to_ras() and to_lps() convenience methods.
    """
    with create_dummy_nifti_file() as file_path:
        img = StatefulImage.load(file_path)

        img.to_lps()
        assert img.axcodes == ("L", "P", "S")

        img.to_ras()
        assert img.axcodes == ("R", "A", "S")


def test_to_reference():
    """
    Test reorienting to match a reference image.
    """
    with create_dummy_nifti_file() as file_path:
        img = StatefulImage.load(file_path)

        # Create a reference image with a different orientation
        ref_affine = np.diag([-1, -1, 1, 1])
        ref_img = nib.Nifti1Image(np.zeros((10, 10, 10)), ref_affine)

        img.to_reference(ref_img)
        assert img.axcodes == ("L", "P", "S")


def test_to_reference_stateful_image():
    """
    Test that to_reference raises a TypeError with a StatefulImage.
    """
    with create_dummy_nifti_file() as file_path:
        img = StatefulImage.load(file_path)
        ref_img = StatefulImage.load(file_path)

        with pytest.raises(TypeError,
                           match="Reference object must not be a StatefulImage."):
            img.to_reference(ref_img)


def test_axcodes_properties_tuple():
    """
    Test the axcodes and original_axcodes properties.
    """
    with create_dummy_nifti_file() as file_path:
        img = StatefulImage.load(file_path, to_orientation=("L", "P", "S"))
        assert img.axcodes == ("L", "P", "S")
        assert img.original_axcodes == ("R", "A", "S")


def test_axcodes_properties_string():
    """
    Test the axcodes and original_axcodes properties.
    """
    with create_dummy_nifti_file() as file_path:
        img = StatefulImage.load(file_path, to_orientation="LPS")
        assert img.axcodes == ("L", "P", "S")
        assert img.original_axcodes == ("R", "A", "S")


def test_str_representation():
    """
    Test the string representation of the StatefulImage.
    """
    with create_dummy_nifti_file() as file_path:
        img = StatefulImage.load(file_path, to_orientation="LPS")
        s = str(img)
        assert "Original axis codes:    ('R', 'A', 'S')" in s
        assert "Current axis codes:     ('L', 'P', 'S')" in s
        assert "Reoriented from original: True" in s


def test_load_no_reorientation():
    """
    Test that loading without reorientation works as expected.
    """
    with create_dummy_nifti_file() as file_path:
        img = StatefulImage.load(file_path, to_orientation=None)
        assert img.axcodes == ("R", "A", "S")
        assert img.original_axcodes == ("R", "A", "S")


def test_reorient_no_op_tuple():
    """
    Test that reorienting to the same orientation is a no-op.
    """
    with create_dummy_nifti_file(in_lps=True) as file_path:
        img = StatefulImage.load(file_path)
        img.reorient(("R", "A", "S"))
        assert img.axcodes == ("R", "A", "S")


def test_reorient_no_op_string():
    """
    Test that reorienting to the same orientation is a no-op.
    """
    with create_dummy_nifti_file(in_lps=True) as file_path:
        img = StatefulImage.load(file_path)
        img.reorient("RAS")
        assert img.axcodes == ("R", "A", "S")


def test_direct_instantiation():
    """
    Test direct instantiation of StatefulImage.
    """
    with create_dummy_nifti_file() as file_path:
        nii = nib.load(file_path)
        img = StatefulImage(nii.dataobj, nii.affine, nii.header)

        assert img.original_axcodes is None
        assert img.axcodes == ("R", "A", "S")

        # Test that save fails without original orientation information
        with pytest.raises(ValueError):
            img.save("test.nii.gz")

        # Test that reorient_to_original fails
        with pytest.raises(ValueError):
            img.reorient_to_original()


@pytest.mark.parametrize("codes, error_msg", [
    (None, "Axis codes cannot be None."),
    ("INVALID", "Invalid axis code 'N' in target."),
    ("RAR", "Target axis codes must be unique."),
    ("LRR", "Target axis codes must be unique."),
    ("LRA", "Conflicting axis codes 'L' and 'R' in target."),
    ("API", "Conflicting axis codes 'A' and 'P' in target."),
])
def test_stateful_image_bad_axcodes_reorient(codes, error_msg):
    """
    Test that reorienting with invalid axis codes raises a ValueError.
    """
    with create_dummy_nifti_file(filename="dummy.nii.gz", in_lps=True) as filepath:
        stateful_img = StatefulImage.load(filepath)
        with pytest.raises(ValueError, match=error_msg):
            stateful_img.reorient(codes)


@pytest.mark.parametrize("codes, error_msg", [
    ("INVALID", "Target axis codes must be of length 3."),
    ("RAR", "Target axis codes must be unique."),
    ("LRR", "Target axis codes must be unique."),
    ("LRA", "Conflicting axis codes 'L' and 'R' in target."),
    ("API", "Conflicting axis codes 'A' and 'P' in target."),
])
def test_stateful_image_bad_axcodes_load(codes, error_msg):
    """
    Test that loading with invalid axis codes raises a ValueError.
    """
    with create_dummy_nifti_file(filename="dummy.nii.gz", in_lps=True) as filepath:
        with pytest.raises(ValueError, match=error_msg):
            StatefulImage.load(filepath, to_orientation=codes)


@pytest.mark.parametrize("codes", [
    ("R", "A", "S"), "RAS",
    ("L", "P", "S"), "LPS",
    ("A", "R", "S"), "ARS",
    ("L", "P", "I"), "LPI",
    ("S", "P", "L"), "SPL",
])
def test_reorient_valid_codes(codes):
    """
    Test that reorienting with valid codes does not raises a ValueError.
    """
    with create_dummy_nifti_file() as file_path:
        img = StatefulImage.load(file_path)
        img.reorient(codes)


@pytest.mark.parametrize('codes, invalid_code', [
    (("X", "Y", "Z"), "X"),
    (("L", "A", "B"), "B"),
])
def test_reorient_invalid_codes(codes, invalid_code):
    """
    Test that reorienting with invalid codes raises a ValueError.
    """
    with create_dummy_nifti_file() as file_path:
        img = StatefulImage.load(file_path)
        with pytest.raises(ValueError,
                           match=f"Invalid axis code '{invalid_code}' in target."):
            img.reorient(codes)


def test_create_from_preserves_source_dtype_and_copies_header():
    """
    Test that create_from preserves source array dtype in header and copies
    header without mutating the reference.
    """
    with create_dummy_nifti_file() as file_path:
        ref_simg = StatefulImage.load(file_path)
        ref_header = ref_simg.header
        original_ref_dtype = ref_header.get_data_dtype()

        # Create source with different dtype (uint8 vs float32)
        source_data = np.ones((10, 10, 10), dtype=np.uint8)
        new_simg = StatefulImage.create_from(source_data, ref_simg)

        # Header of new image should match source dtype
        assert new_simg.header.get_data_dtype() == np.uint8
        assert new_simg.dataobj.dtype == np.uint8
        # Header must be a distinct copy, not mutated in-place
        assert ref_header.get_data_dtype() == original_ref_dtype
        assert new_simg.header is not ref_header


def test_load_preserves_extra_and_file_map(monkeypatch):
    """
    Test that StatefulImage.load preserves extra and file_map from the loaded
    NIfTI image.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (10, 10, 10)
        affine = np.eye(4)
        data = np.ones(shape, dtype=np.float32)
        img = nib.Nifti1Image(data, affine)

        file_path = os.path.join(tmpdir, "test_extra.nii.gz")
        nib.save(img, file_path)

        orig_nib_load = nib.load

        def mock_load(fname):
            loaded = orig_nib_load(fname)
            loaded.extra['custom_key'] = 'custom_val'
            return loaded

        monkeypatch.setattr(nib, 'load', mock_load)

        loaded_simg = StatefulImage.load(file_path)
        assert loaded_simg.extra.get('custom_key') == 'custom_val'
        assert 'image' in loaded_simg.file_map


def test_load_preserves_extra_and_file_map_when_reoriented(monkeypatch):
    """
    Same as test_load_preserves_extra_and_file_map, but with a non-RAS
    input so an actual reorientation happens: as_reoriented() drops extra/
    file_map on its result, so this only passes if load() reads them from
    the original (pre-reorientation) image.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (10, 10, 10)
        affine = np.diag([-1, 1, 1, 1]).astype(np.float32)  # LAS, not RAS
        data = np.ones(shape, dtype=np.float32)
        img = nib.Nifti1Image(data, affine)

        file_path = os.path.join(tmpdir, "test_extra_reoriented.nii.gz")
        nib.save(img, file_path)

        orig_nib_load = nib.load

        def mock_load(fname):
            loaded = orig_nib_load(fname)
            loaded.extra['custom_key'] = 'custom_val'
            return loaded

        # extra is in-memory-only (never written to the NIfTI file itself),
        # so intercept nib.load() to inject it before StatefulImage.load()
        # runs its reorientation logic.
        monkeypatch.setattr(nib, 'load', mock_load)

        loaded_simg = StatefulImage.load(file_path)
        assert loaded_simg.axcodes[:3] == ('R', 'A', 'S')
        assert loaded_simg.extra.get('custom_key') == 'custom_val'
        assert 'image' in loaded_simg.file_map
