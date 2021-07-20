#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['commit_amico.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_NODDI_priors.py', '--help')
    assert ret.success


def test_execution_commit_amico(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_fa = os.path.join(get_home(), 'commit_amico',
                         'fa.nii.gz')
    in_ad = os.path.join(get_home(), 'commit_amico',
                         'ad.nii.gz')
    in_md = os.path.join(get_home(), 'commit_amico',
                         'md.nii.gz')
    ret = script_runner.run('scil_compute_NODDI_priors.py', in_fa, in_ad, in_md,
                            '--out_txt_1fiber', '1fiber.txt',
                            '--out_mask_1fiber', '1fiber.nii.gz',
                            '--out_txt_ventricles', 'ventricules.txt',
                            '--out_mask_ventricles', 'ventricules.nii.gz')
    assert ret.success
