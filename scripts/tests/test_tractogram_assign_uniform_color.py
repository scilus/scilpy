#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tractometry.zip'])
tmp_dir = tempfile.TemporaryDirectory()

in_bundle = os.path.join(SCILPY_HOME, 'tractometry', 'IFGWM.trk')


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_assign_uniform_color.py',
                            '--help')
    assert ret.success


def test_execution_fill(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    ret = script_runner.run('scil_tractogram_assign_uniform_color.py',
                            in_bundle, '--fill_color', '0x000000',
                            '--out_tractogram', 'colored.trk')
    assert ret.success


def test_execution_dict(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # Create a fake dictionary. Using the other hexadecimal format.
    my_dict = {'IFGWM': '#000000'}
    json_file = 'my_json_dict.json'
    with open(json_file, "w+") as f:
        json.dump(my_dict, f)

    ret = script_runner.run('scil_tractogram_assign_uniform_color.py',
                            in_bundle, '--dict_colors', json_file,
                            '--out_suffix', 'colored')
    assert ret.success
