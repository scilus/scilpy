# -*- coding: utf-8 -*-

import numpy as np
import tempfile
import os


def min_dist_to_centroid(bundle_pts, centroid_pts):
    nb_bundle_points = len(bundle_pts)
    nb_centroid_points = len(centroid_pts)
    total_len = nb_bundle_points*nb_centroid_points

    # bundle_points will be shaped like
    # [[bundle_pt1],  ⸣
    #  [bundle_pt1],  ⸠ → Repeated # of centroid points time
    #  [bundle_pt1],  ⸥
    #  ...
    #  [bundle_ptN],
    #  [bundle_ptN],
    #  [bundle_ptN]]
    with tempfile.TemporaryDirectory() as tmp_path:
        bundle_points = np.memmap(os.path.join(tmp_path, 'bundle_points'),
                                  dtype='float16', mode='w+',
                                  shape=(total_len, 3))
        bundle_points[:] = np.repeat(bundle_pts,
                                     nb_centroid_points,
                                     axis=0)

        # centroid_points will be shaped like
        # [[centroid_pt1],  ⸣
        #  [centroid_pt2],  |
        #  ...              ⸠ → Repeated # of points in bundle times
        #  [centroid_pt20], ⸥
        #  [centroid_pt1],
        #  [centroid_pt2],
        #  ...
        #  [centroid_pt20]]
        centroid_points = np.memmap(os.path.join(tmp_path, 'centroid_points'),
                                    dtype='float16', mode='w+',
                                    shape=(total_len, 3))
        centroid_points[:] = np.tile(centroid_pts, (nb_bundle_points, 1))

        # norm will be shaped like
        # [[bundle_pt1 - centroid_pt1],
        #  [bundle_pt1 - centroid_pt2],
        #  [bundle_pt1 - centroid_pt3],
        #  ...
        #  [bundle_ptN - centroid_pt1]]
        #  [bundle_ptN - centroid_pt2]]
        #  ...
        #  [bundle_ptN - centroid_pt20]]
        norm = np.memmap(os.path.join(tmp_path, 'norm'),
                         dtype='float16', mode='w+',
                         shape=(total_len,))
        norm[:] = np.linalg.norm(bundle_points - centroid_points, axis=1)

        # Reshape so we have the distance to each centroid for each
        # bundle point
        dist_to_centroid = np.memmap(os.path.join(tmp_path, 'dist_to_centroid'),
                                     dtype='float16', mode='w+',
                                     shape=(nb_bundle_points, nb_centroid_points))
        dist_to_centroid[:] = norm.reshape(nb_bundle_points,
                                           nb_centroid_points)

        # Find the closest centroid (label and distance) for each point of the
        # bundle
        min_dist_label = np.argmin(dist_to_centroid, axis=1)
        min_dist = np.amin(dist_to_centroid, axis=1)

    return min_dist_label, min_dist
