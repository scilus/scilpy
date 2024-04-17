#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['filtering.zip'])
tmp_dir = tempfile.TemporaryDirectory()

# toDo. Add an atlas and test option --atlas_roi
in_tractogram = os.path.join(SCILPY_HOME, 'filtering',
                             'bundle_all_1mm_inliers.trk')
in_roi = os.path.join(SCILPY_HOME, 'filtering', 'mask.nii.gz')
in_bdo = os.path.join(SCILPY_HOME, 'filtering', 'sc.bdo')


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_filter_by_roi.py', '--help')
    assert ret.success


def test_execution_filtering(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_tractogram_filter_by_roi.py', in_tractogram,
                            'bundle_1.trk', '--display_counts',
                            '--drawn_roi', in_roi, 'any', 'include',
                            '--bdo', in_bdo, 'any', 'include',
                            '--x_plane', '0', 'either_end', 'exclude',
                            '--y_plane', '0', 'all', 'exclude', '0',
                            '--z_plane', '0', 'either_end', 'exclude', '1',
                            '--save_rejected', 'bundle_1_rejected.trk')
    assert ret.success


def test_execution_filtering_overwrite_distance(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_tractogram_filter_by_roi.py', in_tractogram,
                            'bundle_2.trk', '--display_counts',
                            '--drawn_roi', in_roi, 'any', 'include', '2',
                            '--overwrite_distance', 'any', 'include', '4')
    assert ret.success


def test_execution_filtering_list(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # Write a list of options
    filelist = 'my_filelist.txt'
    with open(filelist, 'w') as f:
        f.write('drawn_roi {} any include\n'.format(in_roi))
        f.write('bdo {} "any" "include"\n'.format(in_bdo))
        f.write("bdo {} 'any' include".format(in_bdo))

    ret = script_runner.run('scil_tractogram_filter_by_roi.py', in_tractogram,
                            'bundle_3.trk', '--display_counts',
                            '--filtering_list', filelist)
    assert ret.success
