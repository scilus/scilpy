import os
import tempfile
import nibabel as nib
import numpy as np

from numpy.testing import (assert_array_equal,
                           assert_equal)

from scilpy.segment.tractogram_from_roi import (_extract_vb_one_bundle,
                                                _extract_ib_one_bundle)
from dipy.io.stateful_tractogram import Space, StatefulTractogram


def test_extract_vb_one_bundle():
    # Testing extraction of VS corresponding to a bundle using
    # an empty tractogram which shouldn't raise any error.
    fake_reference = nib.Nifti1Image(
        np.zeros((10, 10, 10, 1)), affine=np.eye(4))
    # The Space type is not important here
    empty_sft = StatefulTractogram([], fake_reference, Space.RASMM)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_mask1_name = os.path.join(tmp_dir, 'fake_mask1.nii.gz')
        fake_mask2_name = os.path.join(tmp_dir, 'fake_mask2.nii.gz')
        nib.save(nib.Nifti1Image(np.zeros((10, 10, 10)),
                 affine=np.eye(4), dtype=np.int8), fake_mask1_name)
        nib.save(nib.Nifti1Image(np.zeros((10, 10, 10)),
                 affine=np.eye(4), dtype=np.int8), fake_mask2_name)

        vs_ids, wpc_ids, bundle_stats = \
            _extract_vb_one_bundle(empty_sft,
                                   fake_mask1_name,
                                   fake_mask2_name,
                                   None, None, None,
                                   None, None, None, None)
        assert_array_equal(vs_ids, [])
        assert_array_equal(wpc_ids, [])
        assert_equal(bundle_stats["VS"], 0)


def test_extract_ib_one_bundle():
    # Testing extraction of IS corresponding to a bundle using
    # an empty tractogram which shouldn't raise any error.
    fake_reference = nib.Nifti1Image(
        np.zeros((10, 10, 10, 1)), affine=np.eye(4))
    # The Space type is not important here
    empty_sft = StatefulTractogram([], fake_reference, Space.RASMM)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_mask1_name = os.path.join(tmp_dir, 'fake_mask1.nii.gz')
        fake_mask2_name = os.path.join(tmp_dir, 'fake_mask2.nii.gz')
        nib.save(nib.Nifti1Image(np.zeros((10, 10, 10)),
                 affine=np.eye(4), dtype=np.int8), fake_mask1_name)
        nib.save(nib.Nifti1Image(np.zeros((10, 10, 10)),
                 affine=np.eye(4), dtype=np.int8), fake_mask2_name)

        fc_sft, fc_ids = _extract_ib_one_bundle(
            empty_sft, fake_mask1_name, fake_mask2_name, None)

        assert_equal(len(fc_sft), 0)
        assert_array_equal(fc_ids, [])
