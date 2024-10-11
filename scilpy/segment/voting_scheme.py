# -*- coding: utf-8 -*-

from itertools import product, repeat
import json
import logging
import multiprocessing
import os
from time import time
import warnings

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamline import transform_streamlines
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from scipy.sparse import lil_matrix

from scilpy.io.streamlines import streamlines_to_memmap
from scilpy.segment.bundleseg import BundleSeg, get_duration

logger = logging.getLogger("BundleSeg")
global MCT, TCT
MCT, TCT = 4, 15


class VotingScheme(object):
    def __init__(self, config, atlas_directory, transformation,
                 output_directory, minimal_vote_ratio=0.5):
        """
        Parameters
        ----------
        config : dict
            Dictionary containing information relative to bundle recognition.
        atlas_directory : list
            List of all directories to be used as atlas by bsg.
            Must contain all bundles as declared in the config file.
        transformation : numpy.ndarray
            Transformation (4x4) bringing the models into subject space.
        output_directory : str
            Directory name where all files will be saved.
        minimal_vote_ratio : float
            Value for the vote ratio for a streamline to be considered.
            (0 < minimal_vote_ratio < 1)
        multi_parameters : int
            Number of runs bsg will performed.
            Enough parameter choices must be provided.
        """
        self.config = config
        self.minimal_vote_ratio = minimal_vote_ratio

        # Scripts parameters
        if isinstance(atlas_directory, list):
            self.atlas_dir = atlas_directory
        else:
            self.atlas_dir = [atlas_directory]

        self.transformation = transformation
        self.output_directory = output_directory

    def _load_bundles_dictionary(self):
        """
        Using all bundles in the configuration file and the input models
        folders, generate all model filepaths.
        Bundles must exist across all folders.
        """
        bundle_names = []
        bundles_filepath = []
        # Generate all input files based on the config file and model directory
        for key in self.config.keys():
            bundle_names.append(key)
            all_atlas_models = product(self.atlas_dir, [key])
            tmp_list = [os.path.join(tag, bundle)
                        for tag, bundle in all_atlas_models]
            bundles_filepath.append(tmp_list)

        logger.info(f'{len(self.atlas_dir)} sub-model directory were found each '
                    f'with {len(bundle_names)} model bundles')

        model_bundles_dict = {}
        bundle_counts = []
        for i, basename in enumerate(bundle_names):
            count = 0
            for j, filename in enumerate(bundles_filepath[i]):
                if not os.path.isfile(filename):
                    continue
                count += 1
                streamlines = nib.streamlines.load(filename).streamlines
                bundle = transform_streamlines(streamlines,
                                               self.transformation)

                model_bundles_dict[filename] = (self.config[basename],
                                                bundle)
            bundle_counts.append(count)

        if sum(bundle_counts) == 0:
            raise IOError("No model bundles found, check input directory.")

        return model_bundles_dict, bundle_names, bundle_counts

    def _find_max_in_sparse_matrix(self, bundle_id, min_vote, bundles_wise_vote):
        """
        Will find the maximum values of a specific row (bundle_id), make
        sure they are the maximum values across bundles (argmax) and above the
        min_vote threshold. Return the indices respecting all three conditions.
        :param bundle_id, int, indices of the bundles in the csr_matrix.
        :param min_vote, int, minimum value for considering (voting).

        :param bundles_wise_vote, scipy.sparse.csr_matrix,
            bundles-wise sparse matrix use for voting.
        """
        if min_vote == 0:
            streamlines_ids = np.asarray([], dtype=np.uint32)
            return streamlines_ids

        streamlines_ids = np.argwhere(bundles_wise_vote[bundle_id] >= min_vote)
        streamlines_ids = np.asarray(streamlines_ids[:, 1], dtype=np.uint32)

        return streamlines_ids

    def _save_recognized_bundles(self, sft, bundle_names,
                                 bundles_wise_vote, bundles_wise_score,
                                 minimum_vote, extension):
        """
        Will save multiple TRK/TCK file and results.json (contains indices)

        Parameters
        ----------
        sft : StatefulTractogram
            Whole brain tractogram (original to dissect)
        bundle_names : list
            Bundle names as defined in the configuration file.
            Will save the bundle using that filename and the extension.
        bundles_wise_vote : lil_matrix
            Array of vote of shape (nbr_bundles x nbr_streamlines).
        bundles_wise_score : lil_matrix
            Array of score of shape (nbr_bundles x nbr_streamlines).
        minimum_vote : np.ndarray
            Value for the vote ratio for a streamline to be considered.
            (0 < minimal_vote < 1)
        extension : str
            Extension for file saving (TRK or TCK).
        """
        results_dict = {}
        for bundle_id in range(len(bundle_names)):
            streamlines_id = self._find_max_in_sparse_matrix(
                bundle_id,
                minimum_vote[bundle_id],
                bundles_wise_vote)

            logger.info(f'{bundle_names[bundle_id]} final recognition got '
                        f'{len(streamlines_id)} streamlines')

            # All models of the same bundle have the same basename
            basename = os.path.splitext(bundle_names[bundle_id])[0]
            new_sft = sft[streamlines_id.T]
            new_sft.remove_invalid_streamlines()
            save_tractogram(new_sft, os.path.join(self.output_directory,
                                                  basename + extension))

            curr_results_dict = {}
            curr_results_dict['indices'] = streamlines_id.tolist()

            scores = bundles_wise_score[bundle_id,
                                        streamlines_id].toarray().flatten()
            curr_results_dict['scores'] = scores.tolist()
            results_dict[basename] = curr_results_dict

        out_logfile = os.path.join(self.output_directory, 'results.json')
        with open(out_logfile, 'w') as outfile:
            json.dump(results_dict, outfile)

    def __call__(self, input_tractograms_path, nbr_processes=1, seed=None,
                 reference=None):
        """
        Entry point function that generate the 'stack' of commands for
        dispatching and launch them using multiprocessing.

        Parameters
        ----------
        input_tractograms_path : str
            Filepath of the whole brain tractogram to segment.
        nbr_processes : int
            Number of processes used for the parallel bundle recognition.
        seed : int
            Seed for the RandomState.
        """
        global MCT, TCT
        # Load the subject tractogram
        load_timer = time()
        reference = input_tractograms_path[0] if reference is None else reference

        wb_streamlines = []
        for in_tractogram in input_tractograms_path:
            sft = load_tractogram(in_tractogram, reference)
            wb_streamlines.extend(sft.streamlines)

        concat_sft = StatefulTractogram(
            wb_streamlines, reference, space=Space.RASMM)
        wb_streamlines = concat_sft.streamlines
        len_wb_streamlines = len(wb_streamlines)

        logger.debug(f'Tractogram {input_tractograms_path} with '
                     f'{len_wb_streamlines} streamlines '
                     f'is loaded in {get_duration(load_timer)} seconds')

        total_timer = time()
        # Each type of bundle is processed separately
        model_bundles_dict, bundle_names, bundle_count = \
            self._load_bundles_dictionary()

        thresholds = [45, 35, 25, TCT]
        rng = np.random.RandomState(seed)
        cluster_timer = time()
        with warnings.catch_warnings(record=True) as _:
            cluster_map = qbx_and_merge(wb_streamlines,
                                        thresholds,
                                        nb_pts=12, rng=rng,
                                        verbose=False)
        clusters_indices = []
        for cluster in cluster_map.clusters:
            clusters_indices.append(cluster.indices)
        centroids = ArraySequence(cluster_map.centroids)
        clusters_indices = ArraySequence(clusters_indices)
        clusters_indices._data = clusters_indices._data.astype(np.uint32)

        logger.info(f'QBx with seed {seed} at {TCT}mm took '
                    f'{get_duration(cluster_timer)}sec. gave '
                    f'{len(cluster_map.centroids)} centroids')

        tmp_dir, tmp_memmap_filenames = streamlines_to_memmap(wb_streamlines,
                                                              'float16')
        bsg = BundleSeg(tmp_memmap_filenames,
                        clusters_indices, centroids)

        # Update all BundleSeg initialisation into a single dictionnary
        pool = multiprocessing.Pool(nbr_processes)
        all_recognized_dict = pool.map(single_recognize,
                                       zip(repeat(bsg),
                                           model_bundles_dict.keys(),
                                           model_bundles_dict.values(),
                                           repeat(bundle_names),
                                           repeat([seed])))
        pool.close()
        pool.join()
        tmp_dir.cleanup()

        logger.info(f'BundleSeg took {get_duration(total_timer)} sec. for '
                    f'{len(bundle_names)} bundles from {len(self.atlas_dir)} atlas')

        save_timer = time()
        bundles_wise_vote = lil_matrix((len(bundle_names),
                                        len_wb_streamlines),
                                       dtype=np.uint8)
        bundles_wise_score = lil_matrix((len(bundle_names),
                                         len_wb_streamlines),
                                        dtype=np.float32)

        for bundle_id, recognized_indices, recognized_scores in all_recognized_dict:
            if recognized_indices is not None:
                if len(recognized_indices) == 0:
                    continue
                tmp_values = bundles_wise_vote[bundle_id, recognized_indices.T]
                bundles_wise_vote[bundle_id, recognized_indices.T] = \
                    tmp_values.toarray() + 1
                tmp_values = bundles_wise_score[bundle_id,
                                                recognized_indices.T]
                bundles_wise_score[bundle_id, recognized_indices.T] = \
                    tmp_values.toarray() + recognized_scores

        bundles_wise_vote = bundles_wise_vote.tocsr()
        bundles_wise_score = bundles_wise_score.tocsr()
        bundles_wise_score[bundles_wise_vote !=
                           0] /= bundles_wise_vote[bundles_wise_vote != 0]

        # Once everything was run, save the results using a voting system
        minimum_vote = np.array(bundle_count) * self.minimal_vote_ratio
        minimum_vote[np.logical_and(minimum_vote > 0, minimum_vote < 1)] = 1
        minimum_vote = minimum_vote.astype(np.uint8)

        _, ext = os.path.splitext(input_tractograms_path[0])

        self._save_recognized_bundles(concat_sft, bundle_names,
                                      bundles_wise_vote,
                                      bundles_wise_score,
                                      minimum_vote, ext)

        logger.info(f'Saving of {len(bundle_names)} files in '
                    f'{self.output_directory} took '
                    f'{get_duration(save_timer)} sec.')


def single_recognize(args):
    """
    Wrapper function to multiprocess recobundles execution.

    Parameters
    ----------
    bsg : Object
        Initialize bsg object with QBx ClusterMap as values
    model_filepath : str
        Path to the model bundle file
    params : tuple
        bundle_pruning_thr : float
            Threshold for pruning the model bundle
        streamlines: ArraySequence
            Streamlines of the model bundle
    bundle_names : list
        List of string with bundle names for models (to get bundle_id)
    seed : int
        Value to initialize the RandomState of numpy

    Returns
    -------
    transf_neighbor : tuple
        bundle_id : (int)
            Unique value to each bundle to identify them.
        recognized_indices : (numpy.ndarray)
            Streamlines indices from the original tractogram.
        recognized_scores : (numpy.ndarray)
            Scores of the recognized streamlines.
    """
    global MCT, TCT
    bsg = args[0]
    model_filepath = args[1]
    bundle_pruning_thr = args[2][0]
    model_bundle = args[2][1]
    bundle_names = args[3]
    np.random.seed(args[4][0])

    # Use for logging and finding the bundle_id
    shorter_tag, ext = os.path.splitext(os.path.basename(model_filepath))

    # Now hardcoded (not useful with FSS from Etienne)
    slr_transform_type = 'similarity'

    recognize_timer = time()
    results = bsg.recognize(model_bundle,
                            model_clust_thr=MCT,
                            pruning_thr=bundle_pruning_thr,
                            slr_transform_type=slr_transform_type,
                            identifier=shorter_tag)
    recognized_indices, recognized_scores = results

    logger.info(f'Model {shorter_tag} recognized {len(recognized_indices)} '
                'streamlines')
    logger.debug(f'Model {model_filepath} with parameters tct={TCT}, mct={MCT}, '
                 f'bpt={bundle_pruning_thr} '
                 f'took {get_duration(recognize_timer)} sec.')

    bundle_id = bundle_names.index(shorter_tag+ext)
    return bundle_id, recognized_indices, recognized_scores
