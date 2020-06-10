#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict())
tmp_dir = tempfile.TemporaryDirectory()


def test_remove_ic(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_tractogram = os.path.join(get_home(), 'bundles',
                                    'bundle_all_1mm.trk')
    ret = script_runner.run('scil_remove_invalid_streamlines.py',
                            input_tractogram, 'bundle_all_1mm.trk', '--cut',
                            '--remove_overlapping', '--remove_single', '-f')
    assert ret.success


def test_single_rb(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_tractogram = os.path.join(get_home(), 'bundles',
                                    'bundle_all_1mm.trk')
    input_model = os.path.join(get_home(), 'bundles', 'fake_atlas',
                               'subj_1', 'bundle_0.tck')
    input_aff = os.path.join(get_home(), 'bundles',
                             'affine.txt')
    input_ref = os.path.join(get_home(), 'bundles',
                             'avg_dwi.nii.gz')
    ret = script_runner.run('scil_recognize_single_bundle.py', input_tractogram,
                            input_model, input_aff, 'bundle_0_reco.tck',
                            '--inverse', '--tractogram_clustering_thr', '12',
                            '--slr_threads', '1', '--out_pickle', 'clusters.pkl',
                            '--reference', input_ref)
    assert ret.success


def test_multi_rb(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_tractogram = os.path.join(get_home(), 'bundles',
                                    'bundle_all_1mm.trk')
    input_conf = os.path.join(get_home(), 'bundles', 'fake_atlas',
                              'default_config_sim.json')
    input_model_1 = os.path.join(get_home(), 'bundles', 'fake_atlas',
                                 'subj_1/')
    input_model_2 = os.path.join(get_home(), 'bundles', 'fake_atlas',
                                 'subj_2/')
    input_model_3 = os.path.join(get_home(), 'bundles', 'fake_atlas',
                                 'subj_3/')
    input_aff = os.path.join(get_home(), 'bundles',
                             'affine.txt')
    ret = script_runner.run('scil_recognize_multi_bundles.py', input_tractogram,
                            input_conf, input_model_1, input_model_2,
                            input_model_3, input_aff, '--inverse',
                            '--tractogram_clustering_thr', '15',
                            '--processes', '1', '--log', 'WARNING')
    assert ret.success


def test_register(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_ref = os.path.join(get_home(), 'bundles',
                             'avg_dwi.nii.gz')
    ret = script_runner.run('scil_register_tractogram.py', 'bundle_0_reco.tck',
                            'voting_results/bundle_0.trk', '--only_rigid',
                            '--moving_tractogram_ref', input_ref)
    assert ret.success


def test_ind_measures(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_ref = os.path.join(get_home(), 'bundles',
                             'avg_dwi.nii.gz')
    ret = script_runner.run('scil_evaluate_bundles_individual_measures.py',
                            'bundle_0_reco.tck', 'voting_results/bundle_0.trk',
                            'AF_L_measures.json', '--reference', input_ref,
                            '--processes', '1')
    assert ret.success


def test_pair_measures(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_ref = os.path.join(get_home(), 'bundles',
                             'avg_dwi.nii.gz')
    ret = script_runner.run('scil_evaluate_bundles_pairwise_agreement_measures.py',
                            'bundle_0_reco.tck', 'voting_results/bundle_0.trk',
                            'AF_L_similarity.json', '--streamline_dice',
                            '--reference', input_ref, '--processes', '1')
    assert ret.success


def test_bin_measures(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    input_tractogram = os.path.join(get_home(), 'bundles',
                                    'bundle_all_1mm.trk')
    input_ref = os.path.join(get_home(), 'bundles',
                             'avg_dwi.nii.gz')
    input_model = os.path.join(get_home(), 'bundles', 'fake_atlas',
                               'subj_1', 'bundle_0.tck')
    ret = script_runner.run('scil_evaluate_bundles_binary_classification_measures.py',
                            'bundle_0_reco.tck', 'voting_results/bundle_0.trk',
                            'AF_L_binary.json', '--streamlines_measures',
                            input_model, input_tractogram, '--processes', '1',
                            '--reference', input_ref)
    assert ret.success
