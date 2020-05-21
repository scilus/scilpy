#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_hdf5_average_density_map.py', '--help')
    assert ret.success
