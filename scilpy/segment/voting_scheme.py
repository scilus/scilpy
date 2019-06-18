#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import product, repeat
import json
import logging
import multiprocessing
import os
import random
from time import time

import nibabel as nib
import numpy as np
from scipy.sparse import dok_matrix

from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamline import transform_streamlines

from scilpy.segment.recobundlesx import RecobundlesX


class VotingScheme(object):
    def __init__(self, config, atlas_directory, transformation,
                 output_directory, minimal_vote_ratio=0.5, multi_parameters=1):
        """
        Parameters
        ----------
        config : dict
            Dictionary containing information relative to bundle recognition
        atlas_directory : list
            List of all directories to be used as atlas by RBx
            Must contain all bundles as declared in the config file 
        transformation : numpy.ndarray
            Transformation (4x4) bringing the models into subject space
        output_directory : str
            Directory name where all files will be saved
        minimal_vote_ratio : float
            Value for the vote ratio for a streamline to be considered 
            (0 < minimal_vote_ratio < 1)
        multi_parameters : int
            Number of runs RBx will performed
            Enough parameter choices must be provided
        """
        self.config = config
        self.multi_parameters = multi_parameters
        self.minimal_vote_ratio = minimal_vote_ratio

        # Scripts parameters
        if isinstance(atlas_directory, list):
            self.atlas_dir = atlas_directory
        else:
            self.atlas_dir = [atlas_directory]

        self.transformation = transformation
        self.output_directory = output_directory

    def _init_bundles_tag(self):
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
            all_atlas_models = list(product(self.atlas_dir, [key]))
            tmp_list = [os.path.join(tag.encode('ascii', 'ignore'),
                                     bundle.encode('ascii', 'ignore'))
                        for tag, bundle in all_atlas_models]
            bundles_filepath.append(tmp_list)

        to_keep = []
        # All models must exist, if not the bundle will be skipped
        for i in range(len(bundles_filepath)):
            missing_count = 0
            missing_files = []
            for j in range(len(bundles_filepath[i])):
                if not os.path.isfile(bundles_filepath[i][j]):
                    missing_count += 1
                    missing_files.append(bundles_filepath[i][j])

            if missing_count == len(bundles_filepath[i]):
                logging.warning("None of the %s exist, this bundle recognition" +
                                " will be skipped", bundle_names[i])
            elif missing_count < len(bundles_filepath[i]) and missing_count > 0:
                logging.error("%s do not exist, this bundle recognition " +
                              "will be skipped", missing_files)
            else:
                to_keep.append(i)

        # Only keep the group of models where all files exist
        bundle_names_exist = [bundle_names[i] for i in to_keep]

        bundles_filepath_exist = [bundles_filepath[i] for i in to_keep]
        logging.info("%s sub-model directory were found each " +
                     "with %s model bundles",
                     len(self.atlas_dir),
                     len(bundle_names_exist))
        logging.debug("The models use for RecobundlesX " +
                      "will be %s", bundles_filepath_exist)

        return bundle_names_exist, bundles_filepath_exist

    def _load_bundles_dictionary(self, bundles_filepath):
        """
        Load all model bundles and store them in a dictionnary where the
        filepaths are the keys and the streamlines the values
        """
        filenames = [filepath for filepath in bundles_filepath]

        model_bundles_dict = {}
        for filename in filenames:
            streamlines = nib.streamlines.load(filename).streamlines
            bundle = transform_streamlines(streamlines, self.transformation)
            model_bundles_dict[filename] = bundle

            logging.debug("Loaded %s with %s streamlines", filename,
                          len(bundle))
            if len(bundle) > 5000:
                logging.warning("%s has above 5000 streamlines", filename)

        return model_bundles_dict

    def _find_max_in_sparse_matrix(self, bundle_id, min_vote,
                                   streamlines_wise_vote, bundles_wise_vote):
        """
        Will find the maximum values of a specific row (bundle_id), make
        sure they are the maximum values across bundles (argmax) and above the
        min_vote threshold. Return the indices respecting all three conditions.
        """ 
        streamlines_indices_in_bundles = []
        streamline_ids = bundles_wise_vote[bundle_id]
        for streamline_id in streamline_ids.keys():
            current_max_vote = -1
            current_arg_max = -1

            for vote_id in streamlines_wise_vote[streamline_id[1]].keys():
                current_vote = streamlines_wise_vote[streamline_id[1], vote_id[1]]
                if current_vote > current_max_vote:
                    current_max_vote = current_vote
                    current_arg_max = vote_id[1]

            if current_arg_max == bundle_id and current_max_vote >= min_vote:
                streamlines_indices_in_bundles.append(streamline_id[1])

        return np.asarray(streamlines_indices_in_bundles, dtype=np.int32)

    def _save_recognized_bundles(self, tractogram, bundle_names,
                                 streamlines_wise_vote, bundles_wise_vote,
                                 minimum_vote, extension):
        """
        Parameters
        ----------
        tractogram : list or ArraySequence
            Filepath of the whole brain tractogram to segment as loaded
            by the nibabel API
        bundle_names : list
            Bundle names as defined in the configuration file
            Will save the bundle using that filename and the extension
        streamlines_wise_vote : dok_matrix
            Array of zeros of shape (nbr_streamlines x nbr_bundles)
        bundles_wise_vote : dok_matrix
            Array of zeros of shape (nbr_bundles x nbr_streamlines)
        minimum_vote : float
            Value for the vote ratio for a streamline to be considered 
            (0 < minimal_vote < 1)
        extension : str
            Extension for file saving (TRK or TCK)

        Will save multiple TRK/TCK file and results.json (contains indices)
        """
        results_dict = {}
        for bundle_id in range(len(bundle_names)):
            streamlines_id = self._find_max_in_sparse_matrix(bundle_id,
                                                             minimum_vote,
                                                             streamlines_wise_vote,
                                                             bundles_wise_vote)

            if len(streamlines_id) < 1:
                logging.error("%s final recognition got %s streamlines",
                              bundle_names[bundle_id], len(streamlines_id))
                continue
            else:
                logging.info("%s final recognition got %s streamlines",
                             bundle_names[bundle_id], len(streamlines_id))

            streamlines = tractogram.streamlines[streamlines_id.T]
            vote_score = streamlines_wise_vote[streamlines_id.T, bundle_id]

            # All models of the same bundle have the same basename
            basename = os.path.join(self.output_directory,
                                    os.path.splitext(bundle_names[bundle_id])[0])
            out_tractogram = nib.streamlines.Tractogram(streamlines,
                                                        affine_to_rasmm=np.eye(4))
            nib.streamlines.save(out_tractogram, basename + extension,
                                 header=tractogram.header)

            curr_results_dict = {}
            curr_results_dict['indices'] = np.asarray(streamlines_id).tolist()
            curr_results_dict['votes'] = np.squeeze(
                vote_score.toarray()).tolist()
            results_dict[basename] = curr_results_dict

        out_logfile = os.path.join(self.output_directory, 'results.json')
        with open(out_logfile, 'w') as outfile:
            json.dump(results_dict, outfile)

    def multi_recognize(self, input_tractogram_path, tractogram_clustering_thr,
                        nb_points=20, nbr_processes=1, seeds=None):
        """
        Parameters
        ----------
        input_tractogram_path : str
            Filepath of the whole brain tractogram to segment
        tractogram_clustering_thr : int
            Distance in mm (for QBx) to cluster the input tractogram
        nb_points : str
            Number of points used for all resampling of streamlines
        nbr_processes : int
            Number of processes used for the parallel bundle recognition
        seeds : list
            List of seed for the RandomState
        """

        # Load the subject tractogram
        timer = time()
        tractogram = nib.streamlines.load(input_tractogram_path)
        wb_streamlines = tractogram.streamlines
        logging.debug("Tractogram %s with %s streamlines " +
                      "is loaded in %s", input_tractogram_path,
                      len(tractogram.streamlines),
                      round(time() - timer, 2))

        # Prepare all tags to read the atlas properly
        bundle_names, bundles_filepath = self._init_bundles_tag()

        # Cluster the whole tractogram only once per possible clustering threshold
        rbx_all = {}
        base_thresholds = [45, 35, 25]
        for seed in seeds:
            rng = np.random.RandomState(seed)
            for clustering_thr in tractogram_clustering_thr:
                timer = time()
                # If necessary, add an extra layer (more optimal)
                if clustering_thr < 15:
                    current_thr_list = base_thresholds + [15, clustering_thr]
                else:
                    current_thr_list = base_thresholds + [clustering_thr]

                cluster_map = qbx_and_merge(wb_streamlines,
                                            current_thr_list,
                                            nb_pts=nb_points, rng=rng,
                                            verbose=False)

                rbx_all[(seed, clustering_thr)] = RecobundlesX(wb_streamlines,
                                                               cluster_map,
                                                               nb_points=nb_points,
                                                               rng=rng)

                logging.info("QBx with seed %s at %smm took %ssec. gave " +
                             "%s centroids", seed, current_thr_list,
                             round(time() - timer, 2),
                             len(cluster_map.centroids))

        total_timer = time()
        processing_dict = {}
        processing_dict['bundle_id'] = []
        processing_dict['tag'] = []
        processing_dict['model_bundle'] = []
        processing_dict['tct'] = []
        processing_dict['mct'] = []
        processing_dict['bpt'] = []
        processing_dict['slr_transform_type'] = []
        processing_dict['seed'] = []

        # Each type of bundle is processed separately
        for seed in seeds:
            for bundle_id in range(len(bundle_names)):
                random.seed(seed)
                bundle_parameters = self.config[bundle_names[bundle_id]]
                model_cluster_thr = bundle_parameters["model_clustering_thr"]
                bundle_pruning_thr = bundle_parameters["bundle_pruning_thr"]
                slr_transform_type = bundle_parameters["slr_transform_type"]
                potential_parameters = list(product(tractogram_clustering_thr,
                                                    model_cluster_thr,
                                                    bundle_pruning_thr))
                random.shuffle(potential_parameters)

                if self.multi_parameters > len(potential_parameters):
                    logging.error("More multi-parameters executions than " +
                                  "potential parameters")
                    self.multi_parameters = len(potential_parameters)

                # Generate a set of parameters for each run
                picked_parameters = potential_parameters[0:self.multi_parameters]

                logging.debug("Parameters choice for %s, for the %s" +
                              " executions are %s", bundle_names[bundle_id],
                              self.multi_parameters,
                              picked_parameters)

                # Using the tag previously generated, load the appropriate
                # model bundles
                model_bundles_dict = self._load_bundles_dictionary(
                    bundles_filepath[bundle_id])

                # Each run (can) have their unique set of parameters
                for parameters in picked_parameters:
                    tct, mct, bpt = parameters

                    # Each bundle (can) have multiple models
                    for tag in bundles_filepath[bundle_id]:
                        model_bundle = model_bundles_dict[tag]
                        processing_dict['bundle_id'] += [bundle_id]
                        processing_dict['tag'] += [tag]
                        processing_dict['model_bundle'] += [model_bundle]
                        processing_dict['tct'] += [tct]
                        processing_dict['mct'] += [mct]
                        processing_dict['bpt'] += [bpt]
                        processing_dict['slr_transform_type'] += [slr_transform_type]
                        processing_dict['seed'] += [seed]

        pool = multiprocessing.Pool(nbr_processes)
        all_measures_dict = pool.map(single_recognize,
                                     zip(repeat(rbx_all),
                                         processing_dict['bundle_id'],
                                         processing_dict['tag'],
                                         processing_dict['model_bundle'],
                                         processing_dict['tct'],
                                         processing_dict['mct'],
                                         processing_dict['bpt'],
                                         processing_dict['slr_transform_type'],
                                         processing_dict['seed']))
        pool.close()
        pool.join()

        streamlines_wise_vote = dok_matrix((len(wb_streamlines),
                                            len(bundle_names)),
                                           dtype=np.int16)
        bundles_wise_vote = dok_matrix((len(bundle_names),
                                        len(wb_streamlines)),
                                       dtype=np.int16)

        for bundle_id, recognized_indices in all_measures_dict:
            if recognized_indices is not None:
                streamlines_wise_vote[recognized_indices.T, bundle_id] += 1
                bundles_wise_vote[bundle_id, recognized_indices.T] += 1

        nb_exec = len(self.atlas_dir) * self.multi_parameters * len(seeds) * \
            len(bundle_names)
        logging.info("RBx took %s sec. for a total of " +
                     "%s exectutions", round(time() - total_timer, 2),
                     nb_exec)
        logging.debug("%s tractogram clustering, %s seeds, " +
                      "%s multi-parameters, %s sub-model directory, " +
                      "%s bundles",
                      len(tractogram_clustering_thr), len(seeds),
                      self.multi_parameters,
                      len(self.atlas_dir),
                      len(bundle_names))

        # Once everything was run, save the results using a voting system
        minimum_vote = round(len(self.atlas_dir) * self.multi_parameters *
                             len(seeds) * self.minimal_vote_ratio)
        minimum_vote = max(minimum_vote, 1)

        extension = os.path.splitext(input_tractogram_path)[1]
        self._save_recognized_bundles(tractogram, bundle_names,
                                      streamlines_wise_vote,
                                      bundles_wise_vote,
                                      minimum_vote, extension)


def single_recognize(args):
    """
    Parameters
    ----------
    rbx_all : dict
        Dictionary with int as key and QBx ClusterMap as values
    bundle_id : int
        Unique value to each bundle to identify them
    tag : str
        Model bundle filepath for logging
    model_bundle : list or Array Sequence
        Model bundle as loaded by the nibabel API
    tct : int
        Tractogram clustering threshold, distance in mm (for QBx)
    mct : int
        Model clustering threshold, distance in mm (for QBx)
    bpt : int
        Bundle pruning threshold, distance in mm
    slr_transform_type : str
        Define the transformation for the local SLR
        [translation, rigid, similarity, scaling]
    seed : int
        Value to initialize the RandomState of numpy
    Returns
    -------
    transf_neighbor : tuple
        bundle_id (int) 
            Unique value to each bundle to identify them
        recognized_indices (numpy.ndarray) 
            Streamlines indices from the original tractogram
    """
    rbx_all = args[0]
    bundle_id = args[1]
    tag = args[2]
    model_bundle = args[3]
    tct = args[4]
    mct = args[5]
    bpt = args[6]
    slr_transform_type = args[7]
    seed = args[8]

    rbx = rbx_all[(seed, tct)]

    timer = time()
    recognized_bundle = rbx.recognize(model_bundle,
                                      model_clust_thr=mct,
                                      bundle_pruning_thr=bpt,
                                      slr_transform_type=slr_transform_type,
                                      identifier=tag)
    recognized_indices = rbx.get_pruned_indices()

    logging.info("Model %s recognized %s streamlines",
                 tag, len(recognized_bundle))
    logging.debug("Model %s (seed %s) with parameters " +
                  "tct=%s, mct=%s, bpt=%s took %s sec.", tag, seed,
                  tct, mct, bpt, round(time() - timer, 2))
    if recognized_indices is None:
        recognized_indices = []
    return bundle_id, np.asarray(recognized_indices, dtype=np.int)
