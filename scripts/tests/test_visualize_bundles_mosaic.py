#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from PIL import Image
from PIL import ImageChops

from scilpy.io.fetcher import get_testing_files_dict, fetch_data, get_home


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['bundles.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_visualize_bundles_mosaic.py', '--help')
    assert ret.success


def test_image_create(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_vol = os.path.join(
        get_home(), 'bundles', 'fibercup_atlas', 'bundle_all_1mm.nii.gz')

    in_bundle = os.path.join(
        get_home(), 'bundles', 'fibercup_atlas', 'subj_1', 'bundle_0.trk')

    comparison = os.path.join(
        get_home(), 'bundles', 'fibercup_atlas', 'subj_1_bundle_0.png')

    ret = script_runner.run('scil_visualize_bundles_mosaic.py',
                            in_vol, in_bundle, 'out.png')
    assert ret.success

    gt = Image.open(comparison)
    gen = Image.open('out.png')

    diff = ImageChops.difference(gt, gen)

    assert not diff.getbbox()
