#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['anatomical_filtering.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_filter_tractogram_anatomically.py',
                            '--help')
    assert ret.success


def test_execution_filtering(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_tractogram = os.path.join(get_home(), 'anatomical_filtering',
                                 'tractogram_filter_ana.trk')
    in_wmparc = os.path.join(get_home(), 'anatomical_filtering',
                             'wmparc_filter_ana.nii.gz')
    ret = script_runner.run('scil_filter_tractogram_anatomically.py',
                            in_tractogram, in_wmparc,
                            os.path.expanduser(tmp_dir.name),
                            '--minL', '40', '--maxL', '200', '-a', '300',
                            '--processes', '1')
    assert ret.success
