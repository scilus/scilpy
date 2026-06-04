# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np

from scilpy.tests.arrays import fodf_3x3_order8_descoteaux07


def test_help_option(script_runner):
    ret = script_runner.run(['scil_fodf_threshold_by_amplitude', '--help'])
    assert ret.success


def test_execution(script_runner):
    with tempfile.TemporaryDirectory() as tmp_dir:
        in_sh = os.path.join(tmp_dir, 'sh.nii.gz')
        out_mask = os.path.join(tmp_dir, 'mask.nii.gz')

        # Create fake SH file
        affine = np.eye(4)
        img = nib.Nifti1Image(fodf_3x3_order8_descoteaux07.astype(np.float32),
                              affine)
        nib.save(img, in_sh)

        # Run with relative threshold
        ret = script_runner.run(['scil_fodf_threshold_by_amplitude',
                                in_sh, out_mask, '--relative', '0.5',
                                '--sh_basis', 'descoteaux07'])
        assert ret.success
        assert os.path.exists(out_mask)

        # Run with absolute threshold
        ret = script_runner.run(['scil_fodf_threshold_by_amplitude',
                                in_sh, out_mask, '--absolute', '0.1',
                                '--sh_basis', 'descoteaux07', '-f'])
        assert ret.success
        assert os.path.exists(out_mask)

        # Run with both thresholds
        ret = script_runner.run(['scil_fodf_threshold_by_amplitude',
                                 in_sh, out_mask, '--relative', '0.5',
                                 '--absolute', '0.1',
                                 '--sh_basis', 'descoteaux07', '-f'])

        assert ret.success
        assert os.path.exists(out_mask)
