import os
import pytest
import tempfile

import nibabel as nib
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def _reorient(img, axcodes):
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(img.affine),
        nib.orientations.axcodes2ornt(axcodes))
    return img.as_reoriented(transform)


def _dice(a, b):
    return 2 * np.sum(a & b) / (np.sum(a) + np.sum(b))


@pytest.fixture(scope="session")
def las_fodf(tmp_path_factory):
    # The test fODF is RAS, in the default basis (descoteaux07_legacy).
    tmp_path = tmp_path_factory.mktemp("las_fodf_data")
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    out_path = str(tmp_path / 'fodf_las.nii.gz')
    nib.save(_reorient(nib.load(in_fodf), ('L', 'A', 'S')), out_path)
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
def test_execution_any_voxel_order(script_runner, monkeypatch,
                                   las_fodf, tmp_path):
    # A RAS input gives the same labels as the LAS input, saved in RAS.
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    out_dir = str(tmp_path / 'out_ras')
    ret = script_runner.run(['scil_fodf_bundleparc', in_fodf, '-f',
                             '--out_dir', out_dir, '--bundles', 'AF_left'])
    assert ret.success

    ref_dir = str(tmp_path / 'out_las')
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf, '-f',
                             '--out_dir', ref_dir, '--bundles', 'AF_left'])
    assert ret.success

    out_img = nib.load(os.path.join(out_dir, 'AF_left.nii.gz'))
    ref_img = nib.load(os.path.join(ref_dir, 'AF_left.nii.gz'))
    assert nib.orientations.aff2axcodes(out_img.affine) == ('R', 'A', 'S')
    np.testing.assert_allclose(out_img.affine, nib.load(in_fodf).affine)

    out_las = _reorient(out_img, ('L', 'A', 'S'))
    np.testing.assert_allclose(out_las.affine, ref_img.affine)
    np.testing.assert_array_equal(out_las.get_fdata(), ref_img.get_fdata())


@pytest.mark.ml
@pytest.mark.serial
def test_execution_sh_basis(script_runner, monkeypatch, las_fodf, tmp_path):
    # A LPS input in the tournier07 basis, given with --sh_basis, gives the
    # same labels as the LAS input in the default basis.
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    lps_fodf = str(tmp_path / 'fodf_lps.nii.gz')
    nib.save(_reorient(nib.load(in_fodf), ('L', 'P', 'S')), lps_fodf)

    tournier_fodf = str(tmp_path / 'fodf_lps_tournier07.nii.gz')
    ret = script_runner.run(['scil_sh_convert', lps_fodf, tournier_fodf,
                             'descoteaux07_legacy', 'tournier07'])
    assert ret.success

    out_dir = str(tmp_path / 'out_tournier07')
    ret = script_runner.run(['scil_fodf_bundleparc', tournier_fodf, '-f',
                             '--out_dir', out_dir, '--bundles', 'AF_left',
                             '--sh_basis', 'tournier07'])
    assert ret.success

    ref_dir = str(tmp_path / 'out_ref')
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf, '-f',
                             '--out_dir', ref_dir, '--bundles', 'AF_left'])
    assert ret.success

    out_img = nib.load(os.path.join(out_dir, 'AF_left.nii.gz'))
    ref_img = nib.load(os.path.join(ref_dir, 'AF_left.nii.gz'))
    assert nib.orientations.aff2axcodes(out_img.affine) == ('L', 'P', 'S')

    # The basis conversion is a refit, so allow a few voxels to change.
    out_las = _reorient(out_img, ('L', 'A', 'S'))
    assert _dice(out_las.get_fdata() > 0, ref_img.get_fdata() > 0) > 0.99


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


def test_execution_3d_volume_error(tmp_path, script_runner):
    in_3d = str(tmp_path / 'volume_3d.nii.gz')
    nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.float32),
                             np.eye(4)), in_3d)

    ret = script_runner.run(['scil_fodf_bundleparc', in_3d,
                             '--bundles', 'FX_left', '-f'])
    assert not ret.success
    assert "must be 4D" in ret.stderr
