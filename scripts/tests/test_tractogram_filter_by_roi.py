#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['filtering.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_filter_by_roi.py', '--help')
    assert ret.success


def test_execution_filtering(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(get_home(), 'filtering',
                                 'bundle_all_1mm_inliers.trk')
    in_roi = os.path.join(get_home(), 'filtering',
                          'mask.nii.gz')
    in_bdo = os.path.join(get_home(), 'filtering',
                          'sc.bdo')
    ret = script_runner.run('scil_tractogram_filter_by_roi.py', in_tractogram,
                            'bundle_4.trk', '--display_counts',
                            '--drawn_roi', in_roi, 'any', 'include',
                            '--bdo', in_bdo, 'any', 'include',
                            '--save_rejected', 'bundle_4_rejected.trk')
    assert ret.success


def test_execution_filtering_distance(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(get_home(), 'filtering',
                                 'bundle_all_1mm_inliers.trk')
    in_roi = os.path.join(get_home(), 'filtering',
                          'mask.nii.gz')
    in_bdo = os.path.join(get_home(), 'filtering',
                          'sc.bdo')
    ret = script_runner.run('scil_tractogram_filter_by_roi.py', in_tractogram,
                            'bundle_5.trk', '--display_counts',
                            '--drawn_roi', in_roi, 'any', 'include', '2',
                            '--bdo', in_bdo, 'any', 'include', '3',
                            '--overwrite_distance', 'any', 'include', '2',
                            '--save_rejected', 'bundle_5_rejected.trk')
    assert ret.success
