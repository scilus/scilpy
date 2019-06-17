#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dipy.align.imaffine import AffineMap
import nibabel as nib
import numpy as np

from scilpy.io.image import get_reference_info
from scilpy.utils.nibabel_tools import get_data


def transform_anatomy(transfo, reference, moving, filename_to_save):
    dim, grid2world = get_reference_info(reference)
    static_data = get_data(reference)

    moving_data, nib_file = get_data(moving, return_object=True)
    moving_affine = nib_file.affine

    if moving_data.ndim == 3 and isinstance(moving_data[0, 0, 0], np.ScalarType):
        orig_type = moving_data.dtype
        affine_map = AffineMap(np.linalg.inv(transfo),
                               dim, grid2world,
                               moving_data.shape, moving_affine)
        resampled = affine_map.transform(moving_data.astype(np.float64))
        nib.save(nib.Nifti1Image(resampled.astype(orig_type), grid2world),
                 filename_to_save)
    elif len(moving_data[0, 0, 0]) > 1:
        if isinstance(moving_data[0, 0, 0], np.void):
            raise ValueError('Does not support TrackVis RGB')

        affine_map = AffineMap(np.linalg.inv(transfo),
                               dim[0:3], grid2world,
                               moving_data.shape[0:3], moving_affine)

        orig_type = moving_data.dtype
        resampled = transform_dwi(affine_map, static_data, moving_data)
        nib.save(nib.Nifti1Image(resampled.astype(orig_type), grid2world),
                 filename_to_save)
    else:
        raise ValueError('Does not support this dataset (shape, type, etc)')
