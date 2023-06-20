# -*- coding: utf-8 -*-

from itertools import product, repeat
import json
import logging
import multiprocessing
import os
from time import time
import warnings

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamline import transform_streamlines
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from scipy.sparse import lil_matrix

from scilpy.io.streamlines import streamlines_to_memmap
from scilpy.segment.recobundlesx import RecobundlesX


class VotingScheme(object):
    def __init__(self, config, atlas_directory, transformation,
                 output_directory, minimal_vote_ratio=0.5):
        """
        Parameters
        ----------
        config : dict
            Dictionary containing information relative to bundle recognition.
        atlas_directory : list
            List of all directories to be used as atlas by RBx.
            Must contain all bundles as declared in the config file.
        transformation : numpy.ndarray
            Transformation (4x4) bringing the models into subject space.
        output_directory : str
            Directory name where all files will be saved.
        minimal_vote_ratio : float
            Value for the vote ratio for a streamline to be considered.
            (0 < minimal_vote_ratio < 1)
        multi_parameters : int
            Number of runs RBx will performed.
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

        logging.info('{0} sub-model directory were found each '
                     'with {1} model bundles'.format(
                         len(self.atlas_dir),
                         len(bundle_names)))

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
        :param bundle_id, int, indices of the bundles in the lil_matrix.
        :param min_vote, int, minimum value for considering (voting).
        :param bundles_wise_vote, lil_matrix, bundles-wise sparse matrix
            use for voting.
        """
        if min_vote == 0:
            streamlines_ids = np.asarray([], dtype=np.uint32)
            # vote_score = np.asarray([], dtype=np.uint32)
            return streamlines_ids  # , vote_score

        streamlines_ids = np.argwhere(bundles_wise_vote[bundle_id] >= min_vote)
        streamlines_ids = np.asarray(streamlines_ids[:, 1], dtype=np.uint32)

        # vote_score = bundles_wise_vote.T[streamlines_ids].tocsr()[:, bundle_id]
        # vote_score = np.squeeze(vote_score.toarray().astype(np.uint32).T)

        return streamlines_ids  # , vote_score

    def _save_recognized_bundles(self, sft, bundle_names,
                                 bundles_wise_vote,
                                 minimum_vote, extension):
        """
        Will save multiple TRK/TCK file and results.json (contains indices)

        Parameters
        ----------
        sft : TODO
        bundle_names : list
            Bundle names as defined in the configuration file.
            Will save the bundle using that filename and the extension.
        bundles_wise_vote : lil_matrix
            Array of zeros of shape (nbr_bundles x nbr_streamlines).
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

            if not streamlines_id.size:
                logging.error('{0} final recognition got {1} streamlines'.format(
                              bundle_names[bundle_id], len(streamlines_id)))
                continue
            else:
                logging.info('{0} final recognition got {1} streamlines'.format(
                             bundle_names[bundle_id], len(streamlines_id)))

            # All models of the same bundle have the same basename
            basename = os.path.join(self.output_directory,
                                    os.path.splitext(bundle_names[bundle_id])[0])
            new_sft = sft[streamlines_id.T]
            new_sft.remove_invalid_streamlines()
            save_tractogram(new_sft, basename + extension)

            curr_results_dict = {}
            curr_results_dict['indices'] = streamlines_id.tolist()
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
        concat_sft = None
        reference = 'same' if reference is None else reference

        for tractogram_path in input_tractograms_path:
            sft = load_tractogram(tractogram_path, reference,
                                  bbox_valid_check=False)

            if concat_sft is None:
                concat_sft = sft
            else:
                concat_sft = concat_sft + sft

        wb_streamlines = concat_sft.streamlines
        len_wb_streamlines = len(wb_streamlines)
        logging.debug('Tractogram {0} with {1} streamlines '
                      'is loaded in {2} seconds'.format(input_tractograms_path,
                                                        len_wb_streamlines,
                                                        round(time() -
                                                              load_timer, 2)))
        total_timer = time()
        # Each type of bundle is processed separately
        model_bundles_dict, bundle_names, bundle_count = \
            self._load_bundles_dictionary()

        thresholds = [45, 35, 25, 12]
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

        logging.info('QBx with seed {0} at 12mm took {1}sec. gave '
                     '{2} centroids'.format(seed,
                                            round(time() - cluster_timer, 2),
                                            len(cluster_map.centroids)))

        concat_sft.streamlines._data = concat_sft.streamlines._data.astype(
            'float16')
        tmp_dir, tmp_memmap_filenames = streamlines_to_memmap(wb_streamlines,
                                                              'float16')
        rbx = RecobundlesX(tmp_memmap_filenames,
                           clusters_indices, centroids)

        # Update all RecobundlesX initialisation into a single dictionnary
        pool = multiprocessing.Pool(nbr_processes)
        all_recognized_dict = pool.map(single_recognize,
                                       zip(repeat(rbx),
                                           model_bundles_dict.keys(),
                                           model_bundles_dict.values(),
                                           repeat(bundle_names),
                                           repeat([seed])))
        pool.close()
        pool.join()
        tmp_dir.cleanup()

        logging.info('RBx took {0} sec. for {1} bundles from {2} atlas'.format(
            round(time() - total_timer, 2),
            len(bundle_names),
            len(self.atlas_dir)))

        save_timer = time()
        bundles_wise_vote = lil_matrix((len(bundle_names),
                                        len_wb_streamlines),
                                       dtype=np.uint8)

        for bundle_id, recognized_indices in all_recognized_dict:
            if recognized_indices is not None:
                tmp_values = bundles_wise_vote[bundle_id, recognized_indices.T]
                bundles_wise_vote[bundle_id,
                                  recognized_indices.T] = tmp_values.toarray() + 1
        bundles_wise_vote = bundles_wise_vote.tocsr()

        # Once everything was run, save the results using a voting system
        minimum_vote = np.array(bundle_count) * self.minimal_vote_ratio
        minimum_vote[np.logical_and(minimum_vote > 0, minimum_vote < 1)] = 1
        minimum_vote = minimum_vote.astype(np.uint8)

        _, ext = os.path.splitext(input_tractograms_path[0])
        self._save_recognized_bundles(concat_sft, bundle_names,
                                      bundles_wise_vote,
                                      minimum_vote, ext)

        logging.info('Saving of {0} files in {1} took {2} sec.'.format(
            len(bundle_names),
            self.output_directory,
            round(time() - save_timer, 2)))


def single_recognize(args):
    """
    Wrapper function to multiprocess recobundles execution.

    Parameters
    ----------
    rbx : Object
        Initialize RBx object with QBx ClusterMap as values
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
    """
    rbx = args[0]
    model_filepath = args[1]
    bundle_pruning_thr = args[2][0]
    model_bundle = args[2][1]
    bundle_names = args[3]
    np.random.seed(args[4][0])

    # Use for logging and finding the bundle_id
    shorter_tag, ext = os.path.splitext(os.path.basename(model_filepath))

    # Now hardcoded (not useful with FSS from Etienne)
    mct = 8
    slr_transform_type = 'similarity'

    recognize_timer = time()
    recognized_indices = rbx.recognize(model_bundle,
                                       model_clust_thr=mct,
                                       pruning_thr=bundle_pruning_thr,
                                       slr_transform_type=slr_transform_type,
                                       identifier=shorter_tag)

    logging.info('Model {0} recognized {1} streamlines'.format(
                 shorter_tag, len(recognized_indices)))
    logging.debug('Model {0} with parameters tct=12, mct=8, bpt={1} '
                  'took {2} sec.'.format(model_filepath, bundle_pruning_thr,
                                         round(time() - recognize_timer, 2)))
    if recognized_indices is None:
        recognized_indices = []

    bundle_id = bundle_names.index(shorter_tag+ext)
    return bundle_id, np.asarray(recognized_indices, dtype=int)
