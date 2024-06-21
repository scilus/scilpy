#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import tempfile

from dipy.io.stateful_tractogram import Space, Origin
import h5py

from scilpy import SCILPY_HOME
from scilpy.io.hdf5 import reconstruct_sft_from_hdf5
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['connectivity.zip'])
tmp_dir = tempfile.TemporaryDirectory()
in_h5 = os.path.join(SCILPY_HOME, 'connectivity', 'decompose.h5')


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_convert_trk_to_hdf5.py', '--help')
    assert ret.success


def test_execution_edge_keys(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    ret = script_runner.run('scil_tractogram_convert_hdf5_to_trk.py',
                            in_h5, 'save_trk/', '--edge_keys', '1_10', '1_7')
    assert ret.success

    # Out directory should have 2 files
    out_files = glob.glob('save_trk/*')
    assert len(out_files) == 2

    ret = script_runner.run('scil_tractogram_convert_trk_to_hdf5.py',
                            'save_trk/1_10.trk', 'save_trk/1_7.trk',
                            'two_edges.h5',
                            '--stored_space', 'voxmm',
                            '--stored_origin', 'nifti')
    assert ret.success

    with h5py.File('two_edges.h5', 'r') as hdf5_file:
        all_hdf5_keys = list(hdf5_file.keys())
        assert all_hdf5_keys == ['1_10', '1_7']

        sfts, _ = reconstruct_sft_from_hdf5(hdf5_file, all_hdf5_keys,
                                            space=Space.VOXMM,
                                            origin=Origin.NIFTI)

    assert len(sfts) == 2
    sfts[0].remove_invalid_streamlines()
    sfts[1].remove_invalid_streamlines()

    assert len(sfts[0]) == 340
    assert len(sfts[1]) == 732
