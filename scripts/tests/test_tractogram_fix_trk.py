#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict


fetch_data(get_testing_files_dict(), keys=['bundles.zip'])
tmp_dir = tempfile.TemporaryDirectory()
fiberdir = os.path.join(SCILPY_HOME, 'bundles', 'fibercup_atlas')
in_bundle = os.path.join(fiberdir, 'subj_1', 'bundle_0.trk')
ref = os.path.join(fiberdir, 'bundle_all_1mm.nii.gz')


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_fix_trk.py', '--help')
    assert ret.success


def test_startrack(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # We don't have a real broken trk, but we can still check that the file
    # runs.
    ret = script_runner.run('scil_tractogram_fix_trk.py', in_bundle,
                            'out_fixed_startrack.trk',
                            '--software', 'startrack',
                            '--reference', ref, '--no_bbox_check')
    assert ret.success


def test_dsi(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # We don't have a real broken trk, but we can still check that the file
    # runs. We don't have two volumes, will flip weirdly and not converge, but
    # will run.
    # Not using --auto-crop on fibercup! Could do with better test data.
    ret = script_runner.run('scil_tractogram_fix_trk.py', in_bundle,
                            'out_fixed_dsi.trk', '--software', 'dsi_studio',
                            '--in_dsi_fa', ref, '--in_native_fa', ref,
                            '--save_transfo', 'out_transfo.txt')
    assert ret.success
