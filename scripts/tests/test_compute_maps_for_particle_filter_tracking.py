#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_maps_for_particle_filter_tracking.py', '--help')
    assert ret.success
