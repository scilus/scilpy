#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dipy.tracking.streamline import transform_streamlines
import nibabel as nib
import numpy as np
from scipy import ndimage

def warp_tractogram(streamlines, transfo, deformation_data, source):
    if source == 'ants':
        flip = [-1, -1, 1]
    elif source == 'dipy':
        flip = [1, 1, 1]

    # Because of duplication, an iteration over chunks of points is necessary
    # for a big dataset (especially if not compressed)
    nb_points = len(streamlines._data)
    current_position = 0
    chunk_size = 1000000
    nb_iteration = int(np.ceil(nb_points/chunk_size))
    inv_transfo = np.linalg.inv(transfo)

    while nb_iteration > 0:
        max_position = min(current_position + chunk_size, nb_points)
        streamline = streamlines._data[current_position:max_position]

        # To access the deformation information, we need to go in voxel space
        streamline_vox = transform_streamlines(streamline,
                                               inv_transfo)

        current_streamline_vox = np.array(streamline_vox).T
        current_streamline_vox_list = current_streamline_vox.tolist()

        x_def = ndimage.map_coordinates(deformation_data[..., 0],
                                        current_streamline_vox_list, order=1)
        y_def = ndimage.map_coordinates(deformation_data[..., 1],
                                        current_streamline_vox_list, order=1)
        z_def = ndimage.map_coordinates(deformation_data[..., 2],
                                        current_streamline_vox_list, order=1)

        # ITK is in LPS and nibabel is in RAS, a flip is necessary for ANTs
        final_streamline = np.array([flip[0]*x_def, flip[1]*y_def, flip[2]*z_def])

        # The deformation obtained is in worldSpace
        if source == 'ants':
            final_streamline += np.array(streamline).T
        elif source == 'dipy':
            final_streamline += current_streamline_vox
            # The tractogram need to be brought back in world space to be saved
            final_streamline = transform_streamlines(final_streamline,
                                                     transfo)

        streamlines._data[current_position:max_position] \
            = final_streamline.T
        current_position = max_position
        nb_iteration -= 1