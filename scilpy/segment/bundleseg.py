# -*- coding: utf-8 -*-

import gc
import logging
from time import time
import warnings

from dipy.align.streamlinear import (BundleMinDistanceMetric,
                                     StreamlineLinearRegistration)
from dipy.segment.fss import FastStreamlineSearch, nearest_from_matrix_col
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamline import (select_random_set_of_streamlines,
                                      transform_streamlines)
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from scipy.sparse import vstack

from scilpy.io.streamlines import reconstruct_streamlines_from_memmap
from scilpy.utils import get_duration

logger = logging.getLogger('BundleSeg')


class BundleSeg(object):
    """
    This class is a 'remastered' version of the Dipy Recobundles class.
    Made to be used in sync with the voting_scheme.
    Adapted in many way for HPC processing and increase control over
    parameters and logger.
    However, in essence they do the same thing.
    https://github.com/nipy/dipy/blob/master/dipy/segment/bundles.py

    References
    ----------
    .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
        bundles using local and global streamline-based registration and
        clustering, Neuroimage, 2017.
    """

    def __init__(self, memmap_filenames, transformation,
                 clusters_indices, wb_centroids, rng=None):
        """
        Parameters
        ----------
        memmap_filenames : tuple
            tuple of filenames for the data, offsets and lengths.
        transformation : numpy.ndarray
            Transformation matrix to apply to the model streamlines.
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
        self.transformation = transformation
        self.wb_clusters_indices = clusters_indices
        self.wb_centroids = wb_centroids
        self.rng = rng

        # For memory management
        self.wb_centroids._data = self.wb_centroids._data.astype(np.float16)

        # For declaration outside of init
        self.neighb_centroids = None
        self.neighb_indices = None
        self.model_streamlines = None
        self.model_centroids = None

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
            Identify the current bundle being recognized for the logger.

        Returns
        -------
        clusters : list
            Streamlines that were recognized by Recobundles and these
            parameters.
        """
        self.model_streamlines = model_bundle

        self._cluster_model_bundle(model_clust_thr,
                                   identifier=identifier)

        if self._reduce_search_space(neighbors_reduction_thr=16) == 0:
            if identifier:
                logger.error(f'{identifier} did not find any neighbors in '
                             'the tractogram')
            return [], []

        self._register_model_to_neighb(slr_transform_type=slr_transform_type)

        if self._reduce_search_space(neighbors_reduction_thr=14) == 0:
            if identifier:
                logger.error(f'{identifier} did not find any neighbors in '
                             'the tractogram')
            return [], []
        del self.neighb_centroids, self.model_centroids
        gc.collect()

        pruned_indices, pruned_scores = self.prune_far_from_model(
            pruning_thr=pruning_thr)

        return pruned_indices, pruned_scores

    def _cluster_model_bundle(self, model_clust_thr, identifier):
        """
        Wrapper function to compute QBx for the model and logging information.

        Parameters
        ----------
        model_clust_thr, float, distance in mm for clustering.
        identifier, str, name of the bundle for logger.
        """
        thresholds = [30, 20, 15, model_clust_thr]
        model_cluster_map = qbx_and_merge(self.model_streamlines, thresholds,
                                          nb_pts=12,
                                          rng=self.rng,
                                          verbose=False)

        self.model_centroids = ArraySequence(model_cluster_map.centroids)
        self.model_centroids._data = self.model_centroids._data.astype(
            np.float16)

        len_centroids = len(self.model_centroids)
        if len_centroids > 1000:
            logger.warning(f'Model {identifier} simplified at threshold '
                           f'{model_clust_thr}mm with {len_centroids} centroids')

    def _reduce_search_space(self, neighbors_reduction_thr=18):
        """
        Wrapper function to discard clusters from the tractogram too far from
        the model and logging informations.
        :param neighbors_reduction_thr, float, distance in mm for thresholding
            to discard distant streamlines.
        """

        centroid_matrix = bundles_distances_mdf(self.model_centroids,
                                                self.wb_centroids).astype(np.float16)
        centroid_matrix[centroid_matrix >
                        neighbors_reduction_thr] = np.inf

        mins = np.min(centroid_matrix, axis=0)
        close_clusters_indices = np.array(np.where(mins != np.inf)[0],
                                          dtype=np.uint32)

        self.neighb_indices = []
        for i in close_clusters_indices:
            self.neighb_indices.extend(self.wb_clusters_indices[i])
        self.neighb_indices = np.array(self.neighb_indices, dtype=np.uint32)

        self.neighb_centroids = ArraySequence([self.wb_centroids[i]
                                               for i in close_clusters_indices])
        self.neighb_centroids._data = self.neighb_centroids._data.astype(
            np.float16)

        return self.neighb_indices.size

    def _register_model_to_neighb(self, select_model=250, select_target=250,
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
                                                  select_model, rng=self.rng)
        moving = select_random_set_of_streamlines(self.neighb_centroids,
                                                  select_target, rng=self.rng)

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
            slr = StreamlineLinearRegistration(metric=metric,
                                               method="L-BFGS-B",
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

        # Apply the transformation to the model streamlines
        self.model_streamlines = transform_streamlines(self.model_streamlines,
                                                       np.linalg.inv(slm.matrix))
        self.model_streamlines._data = self.model_streamlines._data.astype(
            np.float16)

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

        # Typically the neighbors is bigger than the model, so we flip the
        # FSS to be more memory efficient
        with warnings.catch_warnings(record=True) as _:
            fss = FastStreamlineSearch(self.model_streamlines,
                                       pruning_thr, resampling=12)

            CHUNK_SIZE = 1000
            dist_mat_list = []
            for chuck_id in range(0, len(neighb_streamlines), CHUNK_SIZE):
                tmp_dist_mat = fss.radius_search(
                    neighb_streamlines[chuck_id:chuck_id+CHUNK_SIZE],
                    pruning_thr)
                tmp_dist_mat.data = tmp_dist_mat.data.astype(np.float16)
                dist_mat_list.append(tmp_dist_mat.copy())
                del tmp_dist_mat
                gc.collect()

            dist_mat = vstack(dist_mat_list)
            for tmp_dist_mat in dist_mat_list:
                del tmp_dist_mat
            gc.collect()
            dist_mat.data = dist_mat.data.astype(np.float16)
            dist_mat = dist_mat.T

        logger.debug(f'Fast search took of dimensions {dist_mat.shape}: '
                     f'{get_duration(t0)} sec.')
        if dist_mat.size == 0 or dist_mat.shape[1] <= 1:
            return [], []

        # Identify the closest neighbors (remove the zeros, not matched)
        np.absolute(dist_mat.data, out=dist_mat.data)
        non_zero_ids, _, scores = nearest_from_matrix_col(dist_mat)

        del dist_mat, neighb_streamlines
        del fss.ref_slines, fss.bin_dict, fss
        gc.collect()

        # If no streamlines were recognized, return an empty array
        if len(non_zero_ids) != 0:
            # Since the neighbors were clustered, a mapping of indices is neccesary
            final_pruned_indices = self.neighb_indices[non_zero_ids].astype(
                np.uint32)
            final_pruned_scores = scores.astype(np.float16)
            return final_pruned_indices, final_pruned_scores
        else:
            return [], []

    def cleanup(self):
        for indices in [self.neighb_indices, self.wb_clusters_indices]:
            if indices is not None:
                del indices
        del self.model_streamlines._data, self.model_streamlines
