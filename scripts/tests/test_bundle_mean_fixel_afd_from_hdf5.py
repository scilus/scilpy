#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_bundle_mean_fixel_afd_from_hdf5.py',
                            '--help')
    assert ret.success


def test_execution_connectivity(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_h5 = os.path.join(get_home(), 'connectivity', 'decompose.h5')
    in_fodf = os.path.join(get_home(), 'connectivity', 'fodf.nii.gz')
    ret = script_runner.run('scil_bundle_mean_fixel_afd_from_hdf5.py',
                            in_h5, in_fodf, 'decompose_afd.nii.gz',
                            '--length_weighting', '--sh_basis', 'descoteaux07',
                            '--processes', '1')
    assert ret.success
