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
        # Using a slightly larger tolerance because TRK/TCK might have some
        # precision loss or 0.5 offset handling differences
        assert np.allclose(orig, loaded, atol=1e-3)


def test_tck_trk_physical_alignment(tmp_path):
    # Rotation 45 deg around X
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c]
    ])
    affine = np.eye(4)
    affine[:3, :3] = R

    img = create_fake_header(affine)
    img_path = str(tmp_path / "ref.nii.gz")
    nib.save(img, img_path)

    vox_streamlines = [np.array([
        [0, 0, 0],
        [1, 2, 3],
        [5, 5, 5]
    ], dtype=float)]

    sft = StatefulTractogram(vox_streamlines, img, Space.VOX)

    trk_path = str(tmp_path / "tracto.trk")
    tck_path = str(tmp_path / "tracto.tck")

    save_tractogram(sft, trk_path)
    save_tractogram(sft, tck_path)

    sft_trk = load_tractogram(trk_path, img_path)
    sft_tck = load_tractogram(tck_path, img_path)

    sft_trk.to_rasmm()
    sft_tck.to_rasmm()

    for s_trk, s_tck in zip(sft_trk.streamlines, sft_tck.streamlines):
        assert np.allclose(s_trk, s_tck, atol=1e-3)


def test_negative_det_alignment(tmp_path):
    # LAS affine (det < 0)
    affine = np.diag([-1, 1, 1, 1])
    # Add some translation to make it more interesting
    affine[:3, 3] = [100, 100, 100]

    img = create_fake_header(affine)
    img_path = str(tmp_path / "ref.nii.gz")
    nib.save(img, img_path)

    vox_streamlines = [np.array([
        [0, 0, 0],
        [1, 2, 3],
        [5, 5, 5]
    ], dtype=float)]

    sft = StatefulTractogram(vox_streamlines, img, Space.VOX)

    trk_path = str(tmp_path / "tracto.trk")
    tck_path = str(tmp_path / "tracto.tck")

    save_tractogram(sft, trk_path)
    save_tractogram(sft, tck_path)

    sft_trk = load_tractogram(trk_path, img_path)
    sft_tck = load_tractogram(tck_path, img_path)

    sft_trk.to_rasmm()
    sft_tck.to_rasmm()

    # Verify they align in RASMM
    for s_trk, s_tck in zip(sft_trk.streamlines, sft_tck.streamlines):
        assert np.allclose(s_trk, s_tck, atol=1e-3)

    # Verify RASMM coordinates manually
    # v_rasmm = R * v_vox + T
    # For LAS: R = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], T = [100, 100, 100]
    # [0,0,0] -> [-1*0+100, 1*0+100, 1*0+100] = [100, 100, 100]
    # [1,2,3] -> [-1*1+100, 1*2+100, 1*3+100] = [99, 102, 103]
    expected_rasmm = [np.array([
        [100, 100, 100],
        [99, 102, 103],
        [95, 105, 105]
    ], dtype=float)]

    for s_trk, s_exp in zip(sft_trk.streamlines, expected_rasmm):
        assert np.allclose(s_trk, s_exp, atol=1e-3)
