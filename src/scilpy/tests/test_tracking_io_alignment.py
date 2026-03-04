import os
import numpy as np
import nibabel as nib
import pytest
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import load_tractogram, save_tractogram
from scilpy.tracking.utils import save_tractogram as scil_save_tractogram

def create_fake_header(affine, shape=(10, 10, 10)):
    data = np.zeros(shape)
    img = nib.Nifti1Image(data, affine)
    return img

@pytest.mark.parametrize("affine_type", ["iso_1mm", "iso_2mm", "aniso", "complex"])
@pytest.mark.parametrize("ext", [".trk", ".tck"])
def test_tracking_io_alignment(tmp_path, affine_type, ext):
    if affine_type == "iso_1mm":
        affine = np.diag([1, 1, 1, 1])
    elif affine_type == "iso_2mm":
        affine = np.diag([2, 2, 2, 1])
    elif affine_type == "aniso":
        affine = np.diag([1, 1, 2, 1])
    elif affine_type == "complex":
        # Rotation 30 deg around Z, scaling, translation
        theta = np.radians(30)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        S = np.diag([1.1, 0.9, 1.2])
        T = np.array([10, -20, 30])
        affine = np.eye(4)
        affine[:3, :3] = R @ S
        affine[:3, 3] = T
    
    img = create_fake_header(affine)
    img_path = str(tmp_path / "ref.nii.gz")
    nib.save(img, img_path)
    
    # Create streamlines in VOXEL space, origin CENTER
    # (0,0,0) to (5,5,5)
    vox_streamlines = [np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [5, 5, 5]
    ], dtype=float)]
    
    # Convert to RASMM for StatefulTractogram
    # StatefulTractogram expects streamlines in RASMM if space is Space.RASMM
    sft = StatefulTractogram(vox_streamlines, img, Space.VOX)
    
    output_path = str(tmp_path / f"tracto{ext}")
    
    # Method 1: Use DIPY save_tractogram (standard)
    save_tractogram(sft, output_path)
    
    # Reload and check
    sft_loaded = load_tractogram(output_path, img_path)
    
    # Check streamlines in VOX space
    sft_loaded.to_vox()
    loaded_vox = sft_loaded.streamlines
    
    assert len(loaded_vox) == len(vox_streamlines)
    for orig, loaded in zip(vox_streamlines, loaded_vox):
        assert np.allclose(orig, loaded, atol=1e-3)

    # Check streamlines in RASMM space
    sft.to_rasmm()
    sft_loaded.to_rasmm()
    for orig, loaded in zip(sft.streamlines, sft_loaded.streamlines):
        assert np.allclose(orig, loaded, atol=1e-3)

@pytest.mark.parametrize("affine_type", ["iso_1mm", "iso_2mm", "aniso", "complex"])
@pytest.mark.parametrize("ext", [".trk", ".tck"])
def test_scil_save_tractogram_alignment(tmp_path, affine_type, ext):
    if affine_type == "iso_1mm":
        affine = np.diag([1, 1, 1, 1])
    elif affine_type == "iso_2mm":
        affine = np.diag([2, 2, 2, 1])
    elif affine_type == "aniso":
        affine = np.diag([1, 1, 2, 1])
    elif affine_type == "complex":
        theta = np.radians(30)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        S = np.diag([1.1, 0.9, 1.2])
        T = np.array([10, -20, 30])
        affine = np.eye(4)
        affine[:3, :3] = R @ S
        affine[:3, 3] = T

    img = create_fake_header(affine)
    img_path = str(tmp_path / "ref.nii.gz")
    nib.save(img, img_path)

    # Create streamlines in VOXEL space, origin CENTER
    vox_streamlines = [np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [5, 5, 5]
    ], dtype=float)]

    # Generator for scil_save_tractogram
    # it yields (streamline, seed)
    # We make it a list so it's re-iterable if needed
    stream_gen_list = [(s.copy(), s[0].copy()) for s in vox_streamlines]

    output_path = str(tmp_path / f"scil_tracto{ext}")
    tracts_format = nib.streamlines.detect_format(output_path)
    
    # scil_save_tractogram(streamlines_generator, tracts_format, ref_img, total_nb_seeds,
    #                    out_tractogram, min_length, max_length, compress, save_seeds, verbose)
    scil_save_tractogram(stream_gen_list, tracts_format, img, len(vox_streamlines),
                         output_path, 0, 1000, None, False, False)

    # Reload and check
    sft_loaded = load_tractogram(output_path, img_path)
    sft_loaded.to_vox()
    loaded_vox = sft_loaded.streamlines

    assert len(loaded_vox) == len(vox_streamlines)
    for orig, loaded in zip(vox_streamlines, loaded_vox):
        # Using a slightly larger tolerance because TRK/TCK might have some precision loss or 0.5 offset handling differences
        assert np.allclose(orig, loaded, atol=1e-3)
