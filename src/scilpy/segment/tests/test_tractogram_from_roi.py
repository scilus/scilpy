import os
import tempfile

from dipy.io.stateful_tractogram import Space, StatefulTractogram, Origin
import nibabel as nib
import numpy as np
from numpy.testing import (assert_array_equal,
                           assert_equal)

from scilpy.segment.tractogram_from_roi import (_extract_vb_one_bundle,
                                                _extract_ib_one_bundle,
                                                segment_tractogram_from_roi)


def test_compute_masks_from_bundles():
    pass

def test_compute_endpoint_masks():
    pass

def test_segment_tractogram_from_roi():

    # Creating an example.
    # This is already pretty complex, we will not test all complex
    # configurations
    # Inventing a volume of size 5, 5, 5
    # Inventing a streamline that goes
    #   - from right to left in y. Starts at y=0, finishes at y=4
    #   - same, reverse
    #   - in vertical direction. Starts at z=0. Finishes at z=4.
    streamline_r_to_l = [[1.1, 0.2, 1.3],
                         [1.1, 2.2, 1.4],
                         [1.1, 3.3, 1.3],
                         [1.1, 4.3, 1.2]]
    streamline_l_to_r = streamline_r_to_l[::-1]
    bundle1_any_mask = np.zeros((5, 5, 5))
    bundle1_any_mask[1, 1, 1] = 1  # No real point there but passes through

    streamline_vertical = [[3.4, 0.9, 3.9],
                           [3.5, 1.8, 3.8],
                           [3.4, 2.7, 3.6],
                           [3.4, 4.7, 3.5]]
    bundle2_all_mask = np.zeros((5, 5, 5))
    bundle2_all_mask[3, :, 3] = 1

    # Preparing data.
    gt_heads = []  # will be prepared below
    gt_tails = []  # Will be prepared below
    bundle_names = ['l-r', 'vertical']
    lengths = [None, None]
    angles = [359, 359]
    orientation_length = [None, None]
    abs_orientation_length = [None, None]
    all_masks = [None, bundle2_all_mask]
    any_masks = [bundle1_any_mask, None]

    # Many things must be saved on disk.
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f'Created temporary directory: {tmpdirname}')

        fake_ref = nib.Nifti1Image(np.zeros((5, 5, 5)), affine=np.eye(4))
        sft = StatefulTractogram([streamline_r_to_l, streamline_l_to_r,
                                  streamline_vertical],
                                 space=Space.VOX, origin=Origin('corner'),
                                 reference=fake_ref)

        def save_img(array):
            img = nib.Nifti1Image(array.astype('uint8'), affine=np.eye(4))
            nib.save(img, filename)

        # Bundle 1 : left to right (streamlines 1 and 2)
        gt_head = np.zeros((5, 5, 5))
        gt_head[:, 0, :] = 1
        filename = os.path.join(tmpdirname, 'bundle1_head.nii.gz')
        save_img(gt_head)
        gt_heads.append(filename)

        gt_heads.append(filename)
        gt_tail = np.zeros((5, 5, 5))
        gt_tail[:, 4, :] = 1
        filename = os.path.join(tmpdirname, 'bundle1_tail.nii.gz')
        save_img(gt_tail)
        gt_tails.append(filename)

        # Bundle 2: vertical (streamline 2)
        gt_head = np.zeros((5, 5, 5))
        gt_head[:, 0, :] = 1
        filename = os.path.join(tmpdirname, 'bundle2_head.nii.gz')
        save_img(gt_head)
        gt_heads.append(filename)

        gt_heads.append(filename)
        gt_tail = np.zeros((5, 5, 5))
        gt_tail[:, 4, :] = 1
        filename = os.path.join(tmpdirname, 'bundle2_tail.nii.gz')
        save_img(gt_tail)
        gt_tails.append(filename)

        (vb_sft_list, wpc_sft_list, ib_sft_list, nc_sft, ib_names,
         bundle_stats) = segment_tractogram_from_roi(
            sft, gt_tails, gt_heads, bundle_names, lengths, angles,
            orientation_length, abs_orientation_length, all_masks, any_masks,
            out_dir=tmpdirname)

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
