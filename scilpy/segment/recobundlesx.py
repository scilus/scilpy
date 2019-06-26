#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from itertools import chain

import numpy as np

from dipy.align.streamlinear import (StreamlineLinearRegistration,
                                     BundleMinDistanceMetric)
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamline import (select_random_set_of_streamlines,
                                      transform_streamlines)


class RecobundlesX(object):
    """
    This class is a 'remastered' version of the Dipy Recobundles class.
    Made to be used in synch with the voting_scheme
    Adapted in many way for HPC processing and increase control over
    parameters and logging.
    However, in essence they do the same thing
    https://github.com/nipy/dipy/blob/master/dipy/segment/bundles.py

    References
    ----------
    .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
        bundles using local and global streamline-based registration and
        clustering, Neuroimage, 2017.
    """

    def __init__(self, streamlines, cluster_map,
                 nb_points=20, slr_num_thread=1, rng=None):
        """
        Parameters
        ----------
        streamlines : list or ArraySequence
            Whole brain tractogram as loaded by the nibabel API
        cluster_map : obj
            Contains the clusters of QuickBundlesX
        nb_points : int
            Number of points used for all resampling of streamlines
        slr_num_thread : int
            Number of threads for SLR
            Should remain 1 for nearly all use-case
        rng : RandomState
            If None then RandomState is initialized internally.
        """
        self.streamlines = streamlines
        self.cluster_map = cluster_map
        self.centroids = self.cluster_map.centroids
        self.rng = rng

        # Parameters
        self.nb_points = nb_points
        self.slr_num_thread = slr_num_thread

        # For declaration outside of init
        self.neighbors_cluster_thr = None
        self.neighb_centroids = None
        self.neighb_streamlines = None
        self.neighb_indices = None
        self.rtransf_cluster_map = None
        self.model_cluster_map = None
        self.model_centroids = None
        self.pruned_streamlines = None
        self.pruned_indices_per_clusters = None

    def recognize(self, model_bundle,
                  model_clust_thr=8, bundle_pruning_thr=8,
                  slr_transform_type='similarity', identifier=None):
        """
        Parameters
        ----------
        model_bundle : list or ArraySequence
            Model bundle as loaded by the nibabel API
        model_clust_thr : obj
            Distance threshold (mm) for model clustering (QBx)
        bundle_pruning_thr : int
            Distance threshold (mm) for the final pruning
        slr_transform_type : str
            Define the transformation for the local SLR
            [translation, rigid, similarity, scaling]
        identifier : str
            Identify the current bundle being recognize for the logging

        Returns
        -------
        clusters : list
            Streamlines that were recognized by Recobundles and these
            parameters
        """
        self._cluster_model_bundle(model_bundle, model_clust_thr,
                                   identifier=identifier)

        if not self._reduce_search_space():
            if identifier:
                logging.error('{0} did not find any neighbors in '
                              'the tractogram'.format(identifier))
            return []

        if self.slr_num_thread > 0:
            transf_streamlines = self._register_neighb_to_model(
                slr_num_thread=self.slr_num_thread,
                slr_transform_type=slr_transform_type)
        else:
            transf_streamlines = self.neighb_streamlines

        self.pruned_indices_per_clusters = self._prune_what_not_in_model(
            transf_streamlines,
            bundle_pruning_thr=bundle_pruning_thr)

        return self.pruned_streamlines

    def _cluster_model_bundle(self, model, model_clust_thr, identifier=None):
        """
        Wrapper function to compute QBx for the model and logging informations
        :param model, list or arraySequence, streamlines to be used as model
        :param model_clust_thr, float, distance in mm for clustering
        :param identifier, str, name of the bundle for logging
        """
        thresholds = [30, 20, 15, model_clust_thr]
        self.model_cluster_map = qbx_and_merge(model, thresholds,
                                               nb_pts=self.nb_points,
                                               rng=self.rng,
                                               verbose=False)
        self.model_centroids = self.model_cluster_map.centroids
        len_centroids = len(self.model_centroids)
        if len_centroids > 1000:
            logging.warning('Model {0} simplified at threshod '
                            '{1}mm with {2} centroids'.format(identifier,
                            str(model_clust_thr),
                            str(len_centroids)))

    def _reduce_search_space(self, neighbors_reduction_thr=18):
        """
        Wrapper function to discard clusters from the tractogram too far from
        the model and logging informations
        :param neighbors_reduction_thr, float, distance in mm for thresholding
            to discard distant streamlines
        """
        centroid_matrix = bundles_distances_mdf(self.model_centroids,
                                                self.centroids)
        centroid_matrix[centroid_matrix >
                        neighbors_reduction_thr] = np.inf

        mins = np.min(centroid_matrix, axis=0)
        close_clusters_indices = list(np.where(mins != np.inf)[0])
        if len(close_clusters_indices) < 1:
            return False

        close_clusters_indices_tuple = []
        # Order all cluster by size (number of streamlines in it)
        for i in close_clusters_indices:
            close_clusters_indices_tuple.append((
                i, self.cluster_map.clusters_sizes()[i]))
        sorted_tuple = sorted(
            close_clusters_indices_tuple, key=lambda size: size[1])[:-1]

        close_clusters_indices = []
        # Only keep the cluster that are big enough
        for j in range(len(sorted_tuple)):
            if sorted_tuple[j][1] >= 10:
                close_clusters_indices.append(sorted_tuple[j][0])

        close_clusters = self.cluster_map[close_clusters_indices]
        self.neighb_streamlines = list(chain(*close_clusters))
        if not close_clusters_indices:
            return False

        self.neighb_centroids = [self.centroids[i]
                                 for i in close_clusters_indices]
        self.neighb_indices = [cluster.indices for cluster in close_clusters]

        return True

    def _register_neighb_to_model(self, slr_num_thread=1,
                                  select_model=1000, select_target=1000,
                                  slr_transform_type='scaling'):
        """
        Parameters
        ----------
        slr_num_thread : int
            Number of threads for SLR
            Should remain 1 for nearly all use-case
        select_model : int
            Maximum number of clusters to select from the model
        select_target : int
            Maximum number of clusters to select from the neighborhood
        slr_transform_type : str
            Define the transformation for the local SLR
            [translation, rigid, similarity, scaling]

        Returns
        -------
        transf_neighbor : list
            The neighborhood clusters transformed into model space
        """
        possible_slr_transform_type = {'translation': 0, 'rigid': 1,
                                       'similarity': 2, 'scaling': 3}
        static = select_random_set_of_streamlines(self.model_centroids,
                                                  select_model, self.rng)
        moving = select_random_set_of_streamlines(self.neighb_centroids,
                                                  select_target, self.rng)

        # Tuple 0,1,2 are the min & max bound in x,y,z for translation
        # Tuple 3,4,5 are the min & max bound in x,y,z for rotation
        # Tuple 6,7,8 are the min & max bound in x,y,z for scaling
        # For uniform scaling (similarity), tuple #6 is enough
        bounds_dof = [(-20, 20), (-20, 20), (-20, 20),
                      (-10, 10), (-10, 10), (-10, 10),
                      (0.8, 1.2), (0.8, 1.2), (0.8, 1.2)]
        metric = BundleMinDistanceMetric(num_threads=slr_num_thread)
        slr_transform_type_id = possible_slr_transform_type[slr_transform_type]
        if slr_transform_type_id >= 0:
            init_transfo_dof = np.zeros(3)
            slr = StreamlineLinearRegistration(metric=metric,
                                               x0=init_transfo_dof,
                                               bounds=bounds_dof[:3],
                                               num_threads=slr_num_thread)
            slm = slr.optimize(static, moving)

        if slr_transform_type_id >= 1:
            init_transfo_dof = np.zeros(6)
            init_transfo_dof[:3] = slm.xopt

            slr = StreamlineLinearRegistration(metric=metric,
                                               x0=init_transfo_dof,
                                               bounds=bounds_dof[:6],
                                               num_threads=slr_num_thread)
            slm = slr.optimize(static, moving)

        if slr_transform_type_id >= 2:
            init_transfo_dof = np.zeros(7)
            init_transfo_dof[:6] = slm.xopt
            init_transfo_dof[6] = 1.

            slr = StreamlineLinearRegistration(metric=metric,
                                               x0=init_transfo_dof,
                                               bounds=bounds_dof[:7],
                                               num_threads=slr_num_thread)
            slm = slr.optimize(static, moving)

        if slr_transform_type_id >= 3:
            init_transfo_dof = np.zeros(9)
            init_transfo_dof[:6] = slm.xopt[:6]
            init_transfo_dof[6:] = np.array((slm.xopt[6],) * 3)

            slr = StreamlineLinearRegistration(metric=metric,
                                               x0=init_transfo_dof,
                                               bounds=bounds_dof[:9],
                                               num_threads=slr_num_thread)
            slm = slr.optimize(static, moving)

        return transform_streamlines(self.neighb_streamlines, slm.matrix)

    def _prune_what_not_in_model(self, neighbors_to_prune,
                                 bundle_pruning_thr=10,
                                 neighbors_cluster_thr=8):
        """
        Wrapper function to prune clusters from the tractogram too far from
        the model
        :param neighbors_to_prune, list or arraySequence, streamlines to prune
        :param bundle_pruning_thr, float, distance in mm for pruning
        :param neighbors_cluster_thr, float, distance in mm for clustering
        """
        # Neighbors can be refined since the search space is smaller
        thresholds = [40, 30, 20, neighbors_cluster_thr]
        self.rtransf_cluster_map = qbx_and_merge(neighbors_to_prune, thresholds,
                                                 nb_pts=self.nb_points,
                                                 rng=self.rng, verbose=False)

        dist_matrix = bundles_distances_mdf(self.model_centroids,
                                            self.rtransf_cluster_map.centroids)
        dist_matrix[np.isnan(dist_matrix)] = np.inf
        dist_matrix[dist_matrix > bundle_pruning_thr] = np.inf
        mins = np.min(dist_matrix, axis=0)

        pruned_clusters = [self.rtransf_cluster_map[i].indices
                           for i in np.where(mins != np.inf)[0]]
        pruned_indices = list(chain(*pruned_clusters))
        pruned_streamlines = [neighbors_to_prune[i] for i in pruned_indices]

        self.pruned_streamlines = pruned_streamlines
        initial_indices = list(chain(*self.neighb_indices))

        # Since the neighbors were clustered, a mapping of indices is neccesary
        final_indices = []
        for i in range(len(pruned_clusters)):
            final_indices.extend([initial_indices[i]
                                  for i in pruned_clusters[i]])

        return final_indices

    def get_pruned_indices(self):
        """
        Public getter for the final indices recognize by the algorithm
        """
        return self.pruned_indices_per_clusters
