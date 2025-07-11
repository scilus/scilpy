# -*- coding: utf-8 -*-

import gc
from itertools import product, repeat
import json
import logging
from multiprocessing import Manager
import multiprocessing
import os
from time import time
import warnings

from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamline import transform_streamlines
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np

from scilpy.io.streamlines import streamlines_to_memmap, \
    reconstruct_streamlines_from_memmap
from scilpy.segment.bundleseg import BundleSeg
from scilpy.utils import get_duration

logger = logging.getLogger('BundleSeg')

# These parameters are leftovers from Recobundles.
# Now with BundleSeg, they do not need to be modified.
global MCT, TCT
MCT, TCT = 4, 12
# TCT means Tractogram Clustering Threshold (mm)
# MCT means Model Clustering Threshold (mm)


class VotingScheme(object):
    def __init__(self, config, atlas_directory, transformation,
                 output_directory, minimal_vote_ratio=0.5):
        """
        Parameters
        ----------
        config : dict
            Dictionary containing information relative to bundle recognition.
        atlas_directory : list
            List of all directories to be used as atlas by BundleSeg.
            Must contain all bundles as declared in the config file.
        transformation : numpy.ndarray
            Transformation (4x4) bringing the models into subject space.
        output_directory : str
            Directory name where all files will be saved.
        minimal_vote_ratio : float
            Value for the vote ratio for a streamline to be considered.
            (0 < minimal_vote_ratio < 1)
        multi_parameters : int
            Number of runs BundleSeg will performed.
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
            for filepath in tmp_list:
                if not os.path.isfile(filepath):
                    basename = os.path.basename(filepath)
                    tmp_list.remove(filepath)
                    bundle_names.remove(basename)

            if len(tmp_list) > 0:
                bundles_filepath.append(tmp_list)

        logger.info(f'{len(self.atlas_dir)} sub-model directories were found. '
                    f'with {len(bundle_names)} model bundles total')

        model_bundles_dict = {}
        bundle_counts = []
        for i, basename in enumerate(bundle_names):
            count = 0
            for j, filename in enumerate(bundles_filepath[i]):
                if not os.path.isfile(filename):
                    continue
                count += 1

                model_bundles_dict[filename] = self.config[basename]
            bundle_counts.append(count)

        if sum(bundle_counts) == 0:
            raise IOError("No model bundles found, check input directory.")

        return model_bundles_dict, bundle_names, bundle_counts

    def _find_max_in_sparse_matrix(self, bundle_id, min_vote,
                                   bundles_wise_vote):
        """
        Will find the maximum values of a specific row (bundle_id), make
        sure they are the maximum values across bundles (argmax) and above the
        min_vote threshold. Return the indices respecting all three conditions.

        Parameters
        ----------
        bundle_id : int
            Indices of the bundles in the csr_matrix.
        min_vote : int
            Minimum value for considering (voting).
        bundles_wise_vote : scipy.sparse.csr_matrix
            bundles-wise sparse matrix use for voting.

        Returns
        -------
        streamlines_ids : numpy.ndarray
            Indices of the streamlines that are above the min_vote
            threshold and are the maximum values across bundles.
        """
        if min_vote == 0:
            streamlines_ids = np.asarray([], dtype=np.uint32)
            return streamlines_ids

        streamlines_ids = np.argwhere(bundles_wise_vote[bundle_id] >= min_vote)
        streamlines_ids = np.asarray(streamlines_ids, dtype=np.uint32)

        return np.squeeze(streamlines_ids)

    def _save_recognized_bundles(self, memmap_filenames, reference,
                                 bundle_names,
                                 bundles_wise_vote, bundles_wise_score,
                                 minimum_vote, extension):
        """
        Will save multiple TRK/TCK file and results.json (contains indices)

        Parameters
        ----------
        memmap_filenames : list
            List of filenames for the memmap files.
        reference : str
            Reference file for the header.
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
            if streamlines_id.ndim == 0:
                continue

            logger.info(f'{bundle_names[bundle_id]} final recognition got '
                        f'{len(streamlines_id)} streamlines')

            # All models of the same bundle have the same basename
            basename = os.path.splitext(bundle_names[bundle_id])[0]
            streamlines = reconstruct_streamlines_from_memmap(
                memmap_filenames, streamlines_id,
                strs_dtype=np.float16)
            if len(streamlines) != len(streamlines_id):
                raise ValueError('Number of streamlines is not equal to the '
                                 'number of streamlines indices')
            new_sft = StatefulTractogram(streamlines, reference, Space.RASMM)

            if len(new_sft):
                _, indices_to_keep = new_sft.remove_invalid_streamlines()
                streamlines_id = streamlines_id[indices_to_keep]
            save_tractogram(new_sft, os.path.join(self.output_directory,
                                                  basename + extension))

            curr_results_dict = {}
            curr_results_dict['indices'] = streamlines_id.tolist()

            scores = bundles_wise_score[bundle_id,
                                        streamlines_id].flatten()
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
        # Load the subject tractogram
        load_timer = time()
        reference = input_tractograms_path[0] if reference is None else reference

        wb_streamlines = ArraySequence()
        for in_tractogram in input_tractograms_path:
            wb_streamlines.extend(
                nib.streamlines.load(in_tractogram).streamlines)
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

        # Memory cleanup (before multiprocessing)
        cluster_map.refdata = None
        for ref in gc.get_referrers(cluster_map) + \
                gc.get_referrers(wb_streamlines):
            if isinstance(ref, ArraySequence):
                del ref._data
            del ref
        del wb_streamlines, cluster_map
        gc.collect()
        # End of memory cleanup

        bsg = BundleSeg(tmp_memmap_filenames, self.transformation,
                        clusters_indices, centroids)

        # Update all BundleSeg initialisation into a single dictionnary
        with Manager() as manager:
            model_bundles_dict = manager.dict(model_bundles_dict)
            pool = multiprocessing.Pool(nbr_processes)
            all_recognized_dict = pool.imap_unordered(
                single_recognize,
                zip(repeat(bsg), model_bundles_dict.keys(),
                    model_bundles_dict.values(),
                    repeat(bundle_names), repeat([seed])))
            pool.close()
            pool.join()

        logger.info(f'BundleSeg took {get_duration(total_timer)} sec. for '
                    f'{len(bundle_names)} bundles from {len(self.atlas_dir)} atlas')

        bundles_wise_vote = np.zeros((len(bundle_names),
                                      len_wb_streamlines),
                                     dtype=np.uint8)
        bundles_wise_score = np.zeros((len(bundle_names),
                                       len_wb_streamlines),
                                      dtype=np.float16)

        for bundle_id, recognized_indices, recognized_scores in all_recognized_dict:
            if recognized_indices is not None:
                if len(recognized_indices) == 0:
                    continue

                bundles_wise_vote[bundle_id, recognized_indices.T] += 1
                bundles_wise_score[bundle_id,
                                   recognized_indices.T] += recognized_scores

        bundles_wise_score[bundles_wise_vote != 0] \
            /= bundles_wise_vote[bundles_wise_vote != 0]

        # Once everything was run, save the results using a voting system
        minimum_vote = np.array(bundle_count) * self.minimal_vote_ratio
        minimum_vote[np.logical_and(minimum_vote > 0, minimum_vote < 1)] = 1
        minimum_vote = minimum_vote.astype(np.uint8)

        _, ext = os.path.splitext(input_tractograms_path[0])

        save_timer = time()
        self._save_recognized_bundles(tmp_memmap_filenames,
                                      reference,
                                      bundle_names,
                                      bundles_wise_vote,
                                      bundles_wise_score,
                                      minimum_vote, ext)
        tmp_dir.cleanup()

        logger.info(f'Saving of {len(bundle_names)} files in '
                    f'{self.output_directory} took '
                    f'{get_duration(save_timer)} sec.')


def single_recognize_parallel(args):
    """Wrapper function to multiprocess recobundles execution."""
    rbx = args[0]
    model_filepath = args[1]
    bundle_pruning_thr, model_bundle = args[2]
    bundle_names = args[3]
    seed = args[4]
    return single_recognize(rbx, model_filepath, model_bundle,
                            bundle_pruning_thr, bundle_names, seed)


def single_recognize(args):
    """
    Recobundle for a single bundle.

    Parameters
    ----------
    bsg : Object
        Initialize BundleSeg object with QBx ClusterMap as values
    model_filepath : str
        Path to the model bundle file
    model_bundle: ArraySequence
        Model bundle.
    bundle_pruning_thr : float
        Threshold for pruning the model bundle
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
    bsg = args[0]
    model_filepath = args[1]
    bundle_pruning_thr = args[2]
    bundle_names = args[3]
    np.random.seed(args[4][0])

    model_bundle = nib.streamlines.load(model_filepath).streamlines
    model_bundle = transform_streamlines(model_bundle,
                                         bsg.transformation)

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
    del model_bundle._data, model_bundle

    logger.info(f'Model {shorter_tag} recognized {len(recognized_indices)} '
                'streamlines')
    logger.debug(f'Model {model_filepath} with parameters tct={TCT}, mct={MCT}, '
                 f'bpt={bundle_pruning_thr} '
                 f'took {get_duration(recognize_timer)} sec.')

    bundle_id = bundle_names.index(shorter_tag+ext)
    return bundle_id, recognized_indices, recognized_scores
