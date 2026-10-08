import os
import pytest
import tempfile

from dipy.data import get_sphere
import nibabel as nib
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.reconst.sh import convert_sh_basis

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


@pytest.fixture(scope="session")
def las_fodf(tmp_path_factory):
    # The test fODF is RAS and descoteaux07_legacy (scilpy default). Make it
    # the input BundleParc expects: LAS voxel order, tournier07 basis.
    tmp_path = tmp_path_factory.mktemp("las_fodf_data")
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    img = nib.load(in_fodf)
    data = convert_sh_basis(img.get_fdata(dtype=np.float32),
                            get_sphere(name='repulsion724').subdivide(n=1),
                            input_basis='descoteaux07',
                            output_basis='tournier07',
                            is_input_legacy=True, is_output_legacy=False,
                            nbr_processes=1)
    img = nib.Nifti1Image(data.astype(np.float32), img.affine, img.header)
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(img.affine),
        nib.orientations.axcodes2ornt(('L', 'A', 'S')))
    out_path = str(tmp_path / 'fodf_las.nii.gz')
    nib.save(img.as_reoriented(transform), out_path)
    return out_path


def test_help_option(script_runner, monkeypatch):
    ret = script_runner.run(['scil_fodf_bundleparc', '--help'])
    assert ret.success


@pytest.mark.ml
@pytest.mark.serial
def test_execution_las(script_runner, monkeypatch, las_fodf, tmp_path):
    out_dir = str(tmp_path / 'out_las')
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf, '-f',
                             '--out_dir', out_dir, '--bundles', 'FX_left'])
    assert ret.success

    out_img = nib.load(os.path.join(out_dir, 'FX_left.nii.gz'))
    assert nib.orientations.aff2axcodes(out_img.affine) == ('L', 'A', 'S')
    assert out_img.shape == nib.load(las_fodf).shape[:3]


@pytest.mark.ml
@pytest.mark.serial
def test_execution_bundles_on_correct_side(script_runner, monkeypatch,
                                           las_fodf, tmp_path):
    # Left bundles must be left of the brain midline and right bundles right
    # of it. Catches SH orientation errors (e.g. a left-right mirror), which
    # the consistency tests between voxel orders cannot see.
    out_dir = str(tmp_path / 'out_side')
    bundles = ['AF_left', 'AF_right', 'CST_left', 'CST_right']
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf, '-f',
                             '--out_dir', out_dir, '--bundles', *bundles])
    assert ret.success

    # LAS: the first voxel index increases towards the left.
    fodf = nib.load(las_fodf).get_fdata(dtype=np.float32)
    brain_center = np.argwhere(np.any(fodf, axis=-1))[:, 0].mean()
    for b in bundles:
        labels = nib.load(os.path.join(out_dir, f'{b}.nii.gz')).get_fdata()
        offset = np.argwhere(labels > 0)[:, 0].mean() - brain_center
        if b.endswith('_left'):
            assert offset > 0, f'{b} is right of the midline ({offset:.1f})'
        else:
            assert offset < 0, f'{b} is left of the midline ({offset:.1f})'


@pytest.mark.ml
@pytest.mark.serial
def test_execution_ras_reoriented_to_las(script_runner, monkeypatch,
                                         las_fodf, tmp_path):
    # A RAS fODF in the scilpy default basis, converted with the steps
    # documented in the help, must give the same labels as a fODF that is
    # natively LAS.
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    tournier_fodf = str(tmp_path / 'fodf_ras_tournier07.nii.gz')
    ret = script_runner.run(['scil_sh_convert', in_fodf, tournier_fodf,
                             'descoteaux07_legacy', 'tournier07'])
    assert ret.success

    reoriented_fodf = str(tmp_path / 'fodf_ras_to_las.nii.gz')
    ret = script_runner.run(['scil_volume_modify_voxel_order', tournier_fodf,
                             reoriented_fodf, '--new_voxel_order=-1,2,3,4'])
    assert ret.success

    out_dir = str(tmp_path / 'out_reoriented')
    ret = script_runner.run(['scil_fodf_bundleparc', reoriented_fodf, '-f',
                             '--out_dir', out_dir, '--bundles', 'FX_left'])
    assert ret.success

    ref_dir = str(tmp_path / 'out_ref')
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf, '-f',
                             '--out_dir', ref_dir, '--bundles', 'FX_left'])
    assert ret.success

    out_img = nib.load(os.path.join(out_dir, 'FX_left.nii.gz'))
    ref_img = nib.load(os.path.join(ref_dir, 'FX_left.nii.gz'))
    assert nib.orientations.aff2axcodes(out_img.affine) == ('L', 'A', 'S')
    np.testing.assert_allclose(out_img.affine, ref_img.affine)
    np.testing.assert_allclose(out_img.get_fdata(), ref_img.get_fdata())


@pytest.mark.ml
@pytest.mark.serial
def test_execution_fix_space_basis_stride(script_runner, monkeypatch,
                                          las_fodf, tmp_path):
    # Simulate a fODF that BundleParc cannot use as is: LPS voxel order,
    # SH relative to the voxel grid instead of world space, and the scilpy
    # default basis (descoteaux07_legacy). Then fix it with the steps
    # documented in the help and check that the labels match the ones
    # obtained from a clean LAS fODF.
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    img = nib.load(in_fodf)
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(img.affine),
        nib.orientations.axcodes2ornt(('L', 'P', 'S')))
    lps_fodf = str(tmp_path / 'fodf_lps.nii.gz')
    nib.save(img.as_reoriented(transform), lps_fodf)

    messy_fodf = str(tmp_path / 'fodf_lps_voxel.nii.gz')
    ret = script_runner.run(['scil_sh_reorient', lps_fodf, messy_fodf,
                             '--to_voxel', '--sh_basis',
                             'descoteaux07_legacy'])
    assert ret.success

    # As is, the input is rejected.
    ret = script_runner.run(['scil_fodf_bundleparc', messy_fodf, '-f',
                             '--out_dir', str(tmp_path / 'out_messy'),
                             '--bundles', 'FX_left'])
    assert not ret.success

    # 1. Space: SH to world space. Must be done before changing the stride,
    #    since voxel-space SH are only meaningful in their original grid.
    world_fodf = str(tmp_path / 'fodf_lps_world.nii.gz')
    ret = script_runner.run(['scil_sh_reorient', messy_fodf, world_fodf,
                             '--to_world', '--sh_basis',
                             'descoteaux07_legacy'])
    assert ret.success

    # 2. Basis: tournier07 (MRtrix convention).
    basis_fodf = str(tmp_path / 'fodf_lps_world_tournier07.nii.gz')
    ret = script_runner.run(['scil_sh_convert', world_fodf, basis_fodf,
                             'descoteaux07_legacy', 'tournier07'])
    assert ret.success

    # 3. Stride: LAS voxel order.
    fixed_fodf = str(tmp_path / 'fodf_las_world_tournier07.nii.gz')
    ret = script_runner.run(['scil_volume_modify_voxel_order', basis_fodf,
                             fixed_fodf, '--new_voxel_order=-1,2,3,4'])
    assert ret.success

    out_dir = str(tmp_path / 'out_fixed')
    ret = script_runner.run(['scil_fodf_bundleparc', fixed_fodf, '-f',
                             '--out_dir', out_dir, '--bundles', 'FX_left'])
    assert ret.success

    ref_dir = str(tmp_path / 'out_ref')
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf, '-f',
                             '--out_dir', ref_dir, '--bundles', 'FX_left'])
    assert ret.success

    out_img = nib.load(os.path.join(out_dir, 'FX_left.nii.gz'))
    ref_img = nib.load(os.path.join(ref_dir, 'FX_left.nii.gz'))
    np.testing.assert_allclose(out_img.affine, ref_img.affine)

    # SH rotation and basis conversion are refits, so allow a few voxels
    # to flip at the mask boundary.
    out_mask = out_img.get_fdata() > 0
    ref_mask = ref_img.get_fdata() > 0
    dice = 2 * np.sum(out_mask & ref_mask) / \
        (np.sum(out_mask) + np.sum(ref_mask))
    assert dice > 0.99


@pytest.mark.ml
@pytest.mark.serial
def test_execution_100_labels(script_runner, monkeypatch, las_fodf):
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf,
                             '--nb_pts', '100', '-f', '--bundles',
                             'IFO_right'])
    assert ret.success


@pytest.mark.ml
@pytest.mark.serial
def test_execution_keep_biggest_blob(script_runner, monkeypatch, las_fodf):
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf,
                             '--keep_biggest_blob', '-f', '--bundles',
                             'CA'])
    assert ret.success


@pytest.mark.ml
@pytest.mark.serial
def test_execution_invalid_bundle(script_runner, monkeypatch, las_fodf):
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf,
                             '-f', '--bundles', 'CC'])
    assert not ret.success


@pytest.mark.ml
@pytest.mark.serial
def test_execution_mm(script_runner, monkeypatch, las_fodf):
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf,
                             '--mm', '10',
                             '--bundles', 'IFO_right', '-f'])
    assert ret.success


@pytest.mark.ml
@pytest.mark.serial
def test_execution_cont(script_runner, monkeypatch, las_fodf):
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf,
                             '--continuous',
                             '--bundles', 'IFO_right', '-f'])
    assert ret.success


def test_execution_non_las_rejection(script_runner):
    # Tracking fodf is in RAS orientation; bundleparc must reject it
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    ret = script_runner.run(['scil_fodf_bundleparc', in_fodf,
                             '--bundles', 'FX_left', '-f'])
    assert not ret.success
    assert "BundleParc expects fODF input in LAS orientation" in ret.stderr
    assert "scil_volume_modify_voxel_order" in ret.stderr


def test_execution_3d_volume_error(tmp_path, script_runner):
    in_3d = str(tmp_path / 'volume_3d.nii.gz')
    nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.float32),
                             np.eye(4)), in_3d)

    ret = script_runner.run(['scil_fodf_bundleparc', in_3d,
                             '--bundles', 'FX_left', '-f'])
    assert not ret.success
    assert "must be 4D" in ret.stderr
