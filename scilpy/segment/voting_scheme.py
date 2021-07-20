# -*- coding: utf-8 -*-

from itertools import product, repeat
import json
import logging
import multiprocessing
import os
import random
from time import time

from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.streamline import transform_streamlines
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from scipy.sparse import lil_matrix

from scilpy.io.streamlines import (streamlines_to_memmap,
                                   reconstruct_streamlines_from_memmap)
from scilpy.segment.recobundlesx import RecobundlesX


class VotingScheme(object):
    def __init__(self, config, atlas_directory, transformation,
                 output_directory, tractogram_clustering_thr, nb_points=12,
                 minimal_vote_ratio=0.5, multi_parameters=1):
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
        tractogram_clustering_thr : int
            Distance in mm (for QBx) to cluster the input tractogram.
        nb_points : int
            Number of points used for all resampling of streamlines.
        minimal_vote_ratio : float
            Value for the vote ratio for a streamline to be considered.
            (0 < minimal_vote_ratio < 1)
        multi_parameters : int
            Number of runs RBx will performed.
            Enough parameter choices must be provided.
        """
        self.config = config
        self.tractogram_clustering_thr = tractogram_clustering_thr
        self.nb_points = nb_points
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
            all_atlas_models = product(self.atlas_dir, [key])
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
                logging.warning('None of the {0} exist, this bundle'
                                ' will be skipped'.format(bundle_names[i]))
            elif missing_count < len(bundles_filepath[i]) and missing_count > 0:
                logging.error('{0} do not exist, this bundle '
                              'will be skipped'.format(missing_files))
            else:
                to_keep.append(i)

        # Only keep the group of models where all files exist
        bundle_names_exist = [bundle_names[i] for i in to_keep]
        bundles_filepath_exist = [bundles_filepath[i] for i in to_keep]
        logging.info('{0} sub-model directory were found each '
                     'with {1} model bundles'.format(
                         len(self.atlas_dir),
                         len(bundle_names_exist)))
        if len(bundle_names_exist) == 0:
            raise IOError("No model bundles found, check input directory.")

        return bundle_names_exist, bundles_filepath_exist

    def _load_bundles_dictionary(self, bundles_filepath):
        """
        Load all model bundles and store them in a dictionnary where the
        filepaths are the keys and the streamlines the values.
        :param bundles_filepath, list, list of filepaths of model bundles.
        """
        filenames = [filepath for filepath in bundles_filepath]

        model_bundles_dict = {}
        for filename in filenames:
            streamlines = nib.streamlines.load(filename).streamlines
            bundle = transform_streamlines(streamlines, self.transformation)
            model_bundles_dict[filename] = bundle

            if len(bundle) > 5000:
                logging.warning(
                    '{0} has above 5000 streamlines'.format(filename))
        return model_bundles_dict

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
        streamlines_ids = np.argwhere(bundles_wise_vote[bundle_id] >= min_vote)
        streamlines_ids = np.asarray(streamlines_ids[:, 1], dtype=np.int32)

        vote_score = bundles_wise_vote.T[streamlines_ids].tocsr()[:, bundle_id]
        vote_score = np.squeeze(vote_score.toarray().astype(np.int32).T)

        return streamlines_ids, vote_score

    def _save_recognized_bundles(self, tractogram, bundle_names,
                                 bundles_wise_vote,
                                 minimum_vote, extension):
        """
        Will save multiple TRK/TCK file and results.json (contains indices)

        Parameters
        ----------
        tractogram : nib.streamlines.tck.TckFile or nib.streamlines.trk.TrkFile
            Nibabel tractogram object.
        bundle_names : list
            Bundle names as defined in the configuration file.
            Will save the bundle using that filename and the extension.
        bundles_wise_vote : lil_matrix
            Array of zeros of shape (nbr_bundles x nbr_streamlines).
        minimum_vote : float
            Value for the vote ratio for a streamline to be considered.
            (0 < minimal_vote < 1)
        extension : str
            Extension for file saving (TRK or TCK).
        """
        results_dict = {}
        for bundle_id in range(len(bundle_names)):
            streamlines_id, vote_score = self._find_max_in_sparse_matrix(
                bundle_id,
                minimum_vote,
                bundles_wise_vote)

            if not streamlines_id.size:
                logging.error('{0} final recognition got {1} streamlines'.format(
                              bundle_names[bundle_id], len(streamlines_id)))
                continue
            else:
                logging.info('{0} final recognition got {1} streamlines'.format(
                             bundle_names[bundle_id], len(streamlines_id)))

            header = tractogram.header
            streamlines = tractogram.streamlines[streamlines_id.T]
            data_per_streamline = tractogram.tractogram.data_per_streamline[streamlines_id.T]
            data_per_point = tractogram.tractogram.data_per_point[streamlines_id.T]

            # All models of the same bundle have the same basename
            basename = os.path.join(self.output_directory,
                                    os.path.splitext(bundle_names[bundle_id])[0])
            out_tractogram = nib.streamlines.Tractogram(
                streamlines,
                data_per_streamline=data_per_streamline,
                data_per_point=data_per_point,
                affine_to_rasmm=np.eye(4))
            nib.streamlines.save(out_tractogram, basename + extension,
                                 header=header)

            curr_results_dict = {}
            curr_results_dict['indices'] = streamlines_id.tolist()
            curr_results_dict['votes'] = vote_score.tolist()
            results_dict[basename] = curr_results_dict

        out_logfile = os.path.join(self.output_directory, 'results.json')
        with open(out_logfile, 'w') as outfile:
            json.dump(results_dict, outfile)

    def __call__(self, input_tractogram_path, nbr_processes=1, seeds=None):
        """
        Entry point function that generate the 'stack' of commands for
        dispatching and launch them using multiprocessing.

        Parameters
        ----------
        input_tractogram_path : str
            Filepath of the whole brain tractogram to segment.
        nbr_processes : int
            Number of processes used for the parallel bundle recognition.
        seeds : list
            List of seed for the RandomState.
        """

        # Load the subject tractogram
        load_timer = time()
        tractogram = nib.streamlines.load(input_tractogram_path)
        wb_streamlines = tractogram.streamlines
        len_wb_streamlines = len(wb_streamlines)
        logging.debug('Tractogram {0} with {1} streamlines '
                      'is loaded in {2} seconds'.format(input_tractogram_path,
                                                        len_wb_streamlines,
                                                        round(time() -
                                                              load_timer, 2)))

        # Prepare all tags to read the atlas properly
        bundle_names, bundles_filepath = self._init_bundles_tag()

        total_timer = time()
        rbx_params_list = []

        # Each type of bundle is processed separately
        model_bundles_dict = {}
        for bundle_id in range(len(bundle_names)):
            # Using the tag previously generated, load the appropriate
            # model bundles
            model_bundles_dict.update(self._load_bundles_dictionary(
                bundles_filepath[bundle_id]))

            # Using multiple seeds will result in a multiplicative factor
            # of the number of executions
            for seed in seeds:
                random.seed(seed)
                bundle_parameters = self.config[bundle_names[bundle_id]]
                model_cluster_thr = bundle_parameters['model_clustering_thr']
                bundle_pruning_thr = bundle_parameters['bundle_pruning_thr']
                slr_transform_type = bundle_parameters['slr_transform_type']
                potential_parameters = list(product(self.tractogram_clustering_thr,
                                                    model_cluster_thr,
                                                    bundle_pruning_thr))
                random.shuffle(potential_parameters)

                if self.multi_parameters > len(potential_parameters):
                    logging.error('More multi-parameters executions than '
                                  'potential parameters, not enough parameter '
                                  'choices for bundle {0}'.format(
                                      bundle_names[bundle_id]))
                    raise ValueError('Multi-parameters option is too high')

                # Generate a set of parameters for each run
                picked_parameters = potential_parameters[0:self.multi_parameters]

                logging.debug('Parameters choice for {0}, for the {1}'
                              ' executions are {2}'.format(
                                  bundle_names[bundle_id],
                                  self.multi_parameters,
                                  picked_parameters))

                # Each run (can) have their unique set of parameters
                for parameters in picked_parameters:
                    tct, mct, bpt = parameters

                    # Each bundle (can) have multiple models
                    for tag in bundles_filepath[bundle_id]:
                        rbx_params_list.append([bundle_id, tag,
                                                tct, mct, bpt,
                                                slr_transform_type, seed])

        tmp_dir, tmp_memmap_filenames = streamlines_to_memmap(wb_streamlines)
        del wb_streamlines
        comb_param_cluster = product(self.tractogram_clustering_thr, seeds)

        # Clustring is now parallelize
        pool = multiprocessing.Pool(nbr_processes)
        all_rbx_dict = pool.map(single_clusterize_and_rbx_init,
                                zip(repeat(tmp_memmap_filenames),
                                    comb_param_cluster,
                                    repeat(self.nb_points)))
        pool.close()
        pool.join()

        # Update all RecobundlesX initialisation into a single dictionnary
        rbx_dict = {}
        for elem in all_rbx_dict:
            rbx_dict.update(elem)
        random.shuffle(rbx_params_list)

        pool = multiprocessing.Pool(nbr_processes)
        all_recognized_dict = pool.map(single_recognize,
                                       zip(repeat(rbx_dict),
                                           repeat(model_bundles_dict),
                                           rbx_params_list))
        pool.close()
        pool.join()
        tmp_dir.cleanup()

        nb_exec = len(self.atlas_dir) * self.multi_parameters * len(seeds) * \
            len(bundle_names)
        logging.info('RBx took {0} sec. for a total of '
                     '{1} executions'.format(round(time() - total_timer, 2),
                                             nb_exec))
        logging.debug('{0} tractogram clustering, {1} seeds, '
                      '{2} multi-parameters, {3} sub-model directory, '
                      '{4} bundles'.format(
                          len(self.tractogram_clustering_thr), len(seeds),
                          self.multi_parameters,
                          len(self.atlas_dir),
                          len(bundle_names)))

        save_timer = time()
        bundles_wise_vote = lil_matrix((len(bundle_names),
                                        len_wb_streamlines),
                                       dtype=np.int16)

        for bundle_id, recognized_indices in all_recognized_dict:
            if recognized_indices is not None:
                tmp_values = bundles_wise_vote[bundle_id, recognized_indices.T]
                bundles_wise_vote[bundle_id,
                                  recognized_indices.T] = tmp_values.toarray() + 1
        bundles_wise_vote = bundles_wise_vote.tocsr()

        # Once everything was run, save the results using a voting system
        minimum_vote = round(len(self.atlas_dir) * self.multi_parameters *
                             len(seeds) * self.minimal_vote_ratio)
        minimum_vote = max(minimum_vote, 1)

        extension = os.path.splitext(input_tractogram_path)[1]
        self._save_recognized_bundles(tractogram, bundle_names,
                                      bundles_wise_vote,
                                      minimum_vote, extension)

        logging.info('Saving of {0} files in {1} took {2} sec.'.format(
            len(bundle_names),
            self.output_directory,
            round(time() - save_timer, 2)))


def single_clusterize_and_rbx_init(args):
    """
    Wrapper function to multiprocess clustering executions and recobundles
    initialisation.

    Parameters
    ----------
    tmp_memmap_filename: tuple (3)
        Temporary filename for the data, offsets and lengths.

    parameters_list : tuple (3)
        clustering_thr : int
            Distance in mm (for QBx) to cluster the input tractogram.
        seed : int
            Value to initialize the RandomState of numpy.
        nb_points : int
            Number of points used for all resampling of streamlines.

    Returns
    -------
    rbx : dict
        Initialisation of the recobundles class using specific parameters.
    """
    tmp_memmap_filename = args[0]
    wb_streamlines = reconstruct_streamlines_from_memmap(tmp_memmap_filename)
    clustering_thr = args[1][0]
    seed = args[1][1]
    nb_points = args[2]

    rbx = {}
    base_thresholds = [45, 35, 25]
    rng = np.random.RandomState(seed)
    cluster_timer = time()
    # If necessary, add an extra layer (more optimal)
    if clustering_thr < 15:
        current_thr_list = base_thresholds + [15, clustering_thr]
    else:
        current_thr_list = base_thresholds + [clustering_thr]

    cluster_map = qbx_and_merge(wb_streamlines,
                                current_thr_list,
                                nb_pts=nb_points, rng=rng,
                                verbose=False)
    clusters_indices = []
    for cluster in cluster_map.clusters:
        clusters_indices.append(cluster.indices)
    centroids = ArraySequence(cluster_map.centroids)
    clusters_indices = ArraySequence(clusters_indices)
    clusters_indices._data = clusters_indices._data.astype(np.int32)

    rbx[(seed, clustering_thr)] = RecobundlesX(tmp_memmap_filename,
                                               clusters_indices, centroids,
                                               nb_points=nb_points,
                                               rng=rng)
    logging.info('QBx with seed {0} at {1}mm took {2}sec. gave '
                 '{3} centroids'.format(seed, current_thr_list,
                                        round(time() - cluster_timer, 2),
                                        len(cluster_map.centroids)))
    return rbx


def single_recognize(args):
    """
    Wrapper function to multiprocess recobundles execution.

    Parameters
    ----------
    rbx_dict : dict
        Dictionary with int as key and QBx ClusterMap as values
    model_dict : dict
        Dictionary with model tag as key and model streamlines as values
    parameters_list : tuple (8)
        bundle_id : int
            Unique value to each bundle to identify them
        tag : str
            Model bundle filepath for logging
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
        bundle_id : (int)
            Unique value to each bundle to identify them.
        recognized_indices : (numpy.ndarray)
            Streamlines indices from the original tractogram.
    """
    rbx_dict = args[0]
    model_dict = args[1]
    bundle_id = args[2][0]
    tag = args[2][1]
    tct = args[2][2]
    mct = args[2][3]
    bpt = args[2][4]
    slr_transform_type = args[2][5]
    seed = args[2][6]
    model_bundle = model_dict[tag]

    # Use the appropriate initialisation of RBx from the provided parameters
    rbx = rbx_dict[(seed, tct)]
    del rbx_dict

    tmp_split = str(tag).split('/')
    shorter_tag = os.path.join(*tmp_split[-3:])

    recognize_timer = time()
    recognized_indices = rbx.recognize(model_bundle,
                                       model_clust_thr=mct,
                                       bundle_pruning_thr=bpt,
                                       slr_transform_type=slr_transform_type,
                                       identifier=shorter_tag)

    logging.info('Model {0} recognized {1} streamlines'.format(
                 shorter_tag, len(recognized_indices)))
    logging.debug('Model {0} (seed {1}) with parameters '
                  'tct={2}, mct={3}, bpt={4} '
                  'took {5} sec.'.format(shorter_tag, seed,
                                         tct, mct, bpt,
                                         round(time() - recognize_timer, 2)))
    if recognized_indices is None:
        recognized_indices = []
    return bundle_id, np.asarray(recognized_indices, dtype=int)
