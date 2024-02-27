# -*- coding: utf-8 -*-

import logging
from time import time
import warnings

from dipy.align.streamlinear import (BundleMinDistanceMetric,
                                     StreamlineLinearRegistration)
from dipy.segment.fss import FastStreamlineSearch
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamline import (select_random_set_of_streamlines,
                                      transform_streamlines)
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np

from scilpy.io.streamlines import reconstruct_streamlines_from_memmap


class RecobundlesX(object):
    """
    This class is a 'remastered' version of the Dipy Recobundles class.
    Made to be used in sync with the voting_scheme.
    Adapted in many way for HPC processing and increase control over
    parameters and logging.
    However, in essence they do the same thing.
    https://github.com/nipy/dipy/blob/master/dipy/segment/bundles.py

    References
    ----------
    .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
        bundles using local and global streamline-based registration and
        clustering, Neuroimage, 2017.
    """

    def __init__(self, memmap_filenames, clusters_indices, wb_centroids,
                 rng=None):
        """
        Parameters
        ----------
        memmap_filenames : tuple
            tuple of filenames for the data, offsets and lengths.
        clusters_indices: ArraySequence
            ArraySequence containing the indices of the streamlines per
            cluster.
        wb_centroids : list of numpy.ndarray
            List contaning the average streamline per cluster as obtained
            from qbx.
        rng : RandomState
            If None then RandomState is initialized internally.
        """
        self.memmap_filenames = memmap_filenames
        self.wb_clusters_indices = clusters_indices
        self.centroids = wb_centroids
        self.rng = rng

        # For declaration outside of init
        self.neighb_centroids = None
        self.neighb_indices = None
        self.models_streamlines = None
        self.model_centroids = None
        self.final_pruned_indices = None

    def recognize(self, model_bundle,
                  model_clust_thr=8, pruning_thr=8,
                  slr_transform_type='similarity', identifier=None):
        """
        Parameters
        ----------
        model_bundle : list or ArraySequence
            Model bundle as loaded by the nibabel API.
        model_clust_thr : obj
            Distance threshold (mm) for model clustering (QBx)
        pruning_thr : int
            Distance threshold (mm) for the final pruning.
        slr_transform_type : str
            Define the transformation for the local SLR.
            [translation, rigid, similarity, scaling]
        identifier : str
            Identify the current bundle being recognized for the logging.

        Returns
        -------
        clusters : list
            Streamlines that were recognized by Recobundles and these
            parameters.
        """

        self.model_streamlines = model_bundle
        self._cluster_model_bundle(model_clust_thr,
                                   identifier=identifier)

        if not self._reduce_search_space(neighbors_reduction_thr=16):
            if identifier:
                logging.error('{0} did not find any neighbors in '
                              'the tractogram'.format(identifier))
            return np.array([], dtype=np.uint32)

        self._register_model_to_neighb(slr_transform_type=slr_transform_type)

        # self._reduce_search_space(neighbors_reduction_thr=12)
        if not self._reduce_search_space(neighbors_reduction_thr=12):
            if identifier:
                logging.error('{0} did not find any neighbors in '
                              'the tractogram'.format(identifier))
            return np.array([], dtype=np.uint32)

        self.prune_far_from_model(pruning_thr=pruning_thr)

        self.cleanup()
        return self.get_final_pruned_indices()

    def _cluster_model_bundle(self, model_clust_thr, identifier):
        """
        Wrapper function to compute QBx for the model and logging information.

        Parameters
        ----------
        model_clust_thr, float, distance in mm for clustering.
        identifier, str, name of the bundle for logging.
        """
        thresholds = [30, 20, 15, model_clust_thr]
        model_cluster_map = qbx_and_merge(self.model_streamlines, thresholds,
                                          nb_pts=12,
                                          rng=self.rng,
                                          verbose=False)

        self.model_centroids = ArraySequence(model_cluster_map.centroids)
        len_centroids = len(self.model_centroids)
        if len_centroids > 1000:
            logging.warning('Model {0} simplified at threshold '
                            '{1}mm with {2} centroids'.format(identifier,
                                                              str(model_clust_thr),
                                                              str(len_centroids)))

    def _reduce_search_space(self, neighbors_reduction_thr=18):
        """
        Wrapper function to discard clusters from the tractogram too far from
        the model and logging informations.
        :param neighbors_reduction_thr, float, distance in mm for thresholding
            to discard distant streamlines.
        """
        centroid_matrix = bundles_distances_mdf(self.model_centroids,
                                                self.centroids).astype(np.float16)
        centroid_matrix[centroid_matrix >
                        neighbors_reduction_thr] = np.inf

        mins = np.min(centroid_matrix, axis=0)
        close_clusters_indices = np.array(np.where(mins != np.inf)[0],
                                          dtype=np.uint32)

        self.neighb_indices = []
        for i in close_clusters_indices:
            self.neighb_indices.extend(self.wb_clusters_indices[i])
        self.neighb_indices = np.array(self.neighb_indices, dtype=np.uint32)

        self.neighb_centroids = [self.centroids[i]
                                 for i in close_clusters_indices]

        return self.neighb_indices.size

    def _register_model_to_neighb(self, select_model=1000, select_target=1000,
                                  slr_transform_type='similarity'):
        """
        Parameters
        ----------
        select_model : int
            Maximum number of clusters to select from the model.
        select_target : int
            Maximum number of clusters to select from the neighborhood.
        slr_transform_type : str
            Define the transformation for the local SLR.
            [translation, rigid, similarity, scaling].

        Returns
        -------
        transf_neighbor : list
            The neighborhood clusters transformed into model space.
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
        metric = BundleMinDistanceMetric(num_threads=1)
        slr_transform_type_id = possible_slr_transform_type[slr_transform_type]
        if slr_transform_type_id >= 0:
            init_transfo_dof = np.zeros(3)
            slr = StreamlineLinearRegistration(metric=metric, method="L-BFGS-B",
                                               x0=init_transfo_dof,
                                               bounds=bounds_dof[:3],
                                               num_threads=1)
            slm = slr.optimize(static, moving)

        if slr_transform_type_id >= 1:
            init_transfo_dof = np.zeros(6)
            init_transfo_dof[:3] = slm.xopt

            slr = StreamlineLinearRegistration(metric=metric,
                                               x0=init_transfo_dof,
                                               bounds=bounds_dof[:6],
                                               num_threads=1)
            slm = slr.optimize(static, moving)

        if slr_transform_type_id >= 2:
            if slr_transform_type_id == 2:
                init_transfo_dof = np.zeros(7)
                init_transfo_dof[:6] = slm.xopt
                init_transfo_dof[6] = 1.

                slr = StreamlineLinearRegistration(metric=metric,
                                                   x0=init_transfo_dof,
                                                   bounds=bounds_dof[:7],
                                                   num_threads=1)
                slm = slr.optimize(static, moving)

            else:
                init_transfo_dof = np.zeros(9)
                init_transfo_dof[:6] = slm.xopt[:6]
                init_transfo_dof[6:] = np.array((slm.xopt[6],) * 3)

                slr = StreamlineLinearRegistration(metric=metric,
                                                   x0=init_transfo_dof,
                                                   bounds=bounds_dof[:9],
                                                   num_threads=1)
                slm = slr.optimize(static, moving)
        self.model_centroids = transform_streamlines(self.model_centroids,
                                                     np.linalg.inv(slm.matrix))

    def prune_far_from_model(self, pruning_thr=10):
        """
        Wrapper function to prune clusters from the tractogram too far from
        the model.

        Parameters
        ----------
        pruning_thr: float, distance in
            thresholds = [32, 16, 24, neighbors_cluster_thr]
        """
        # Neighbors can be refined since the search space is smaller
        t0 = time()

        neighb_streamlines = reconstruct_streamlines_from_memmap(
            self.memmap_filenames, self.neighb_indices, strs_dtype=np.float16)

        with warnings.catch_warnings(record=True) as _:
            fss = FastStreamlineSearch(neighb_streamlines,
                                       pruning_thr, resampling=12)
            dist_mat = fss.radius_search(self.model_streamlines,
                                         pruning_thr)

        logging.debug("Fast search took of dimensions {0}: {1} sec.".format(
            dist_mat.shape, np.round(time() - t0, 2)))

        sparse_dist_mat = np.abs(dist_mat.tocsr())
        sparse_dist_vec = np.squeeze(np.max(sparse_dist_mat, axis=0).toarray())
        pruned_indices = np.where(sparse_dist_vec > 1e-6)[0]

        # Since the neighbors were clustered, a mapping of indices is neccesary
        self.final_pruned_indices = self.neighb_indices[pruned_indices].astype(
            np.uint32)

        return self.final_pruned_indices

    def get_final_pruned_indices(self):
        """
        Public getter for the final indices recognize by the algorithm.
        """
        return self.final_pruned_indices

    def cleanup(self):
        del self.neighb_centroids
        del self.neighb_indices
        del self.model_streamlines
        del self.model_centroids
