#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['anatomical_filtering.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_filter_by_anatomy.py',
                            '--help')
    assert ret.success


def test_execution_filtering_all_options(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'anatomical_filtering',
                                 'tractogram_filter_ana.trk')
    in_wmparc = os.path.join(SCILPY_HOME, 'anatomical_filtering',
                             'wmparc_filter_ana.nii.gz')
    ret = script_runner.run('scil_tractogram_filter_by_anatomy.py',
                            in_tractogram, in_wmparc,
                            os.path.expanduser(tmp_dir.name),
                            '--minL', '40', '--maxL', '200', '-a', '300',
                            '--processes', '1', '--save_volumes',
                            '--ctx_dilation_radius', '2',
                            '--save_intermediate_tractograms',
                            '--save_counts',
                            '--save_rejected',
                            '-f')
    assert ret.success


def test_execution_filtering_rejected(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'anatomical_filtering',
                                 'tractogram_filter_ana.trk')
    in_wmparc = os.path.join(SCILPY_HOME, 'anatomical_filtering',
                             'wmparc_filter_ana.nii.gz')
    ret = script_runner.run('scil_tractogram_filter_by_anatomy.py',
                            in_tractogram, in_wmparc,
                            os.path.expanduser(tmp_dir.name),
                            '--minL', '40', '--maxL', '200', '-a', '300',
                            '--processes', '1', '--save_volumes',
                            '--ctx_dilation_radius', '2',
                            '--save_counts',
                            '--save_rejected',
                            '-f')
    assert ret.success


def test_execution_filtering_save_intermediate(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(SCILPY_HOME, 'anatomical_filtering',
                                 'tractogram_filter_ana.trk')
    in_wmparc = os.path.join(SCILPY_HOME, 'anatomical_filtering',
                             'wmparc_filter_ana.nii.gz')
    ret = script_runner.run('scil_tractogram_filter_by_anatomy.py',
                            in_tractogram, in_wmparc,
                            os.path.expanduser(tmp_dir.name),
                            '--minL', '40', '--maxL', '200', '-a', '300',
                            '--processes', '1', '--save_volumes',
                            '--ctx_dilation_radius', '2',
                            '--save_intermediate_tractograms',
                            '-f')
    assert ret.success
