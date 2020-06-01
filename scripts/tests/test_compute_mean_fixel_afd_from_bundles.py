#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_mean_fixel_afd_from_bundles.py', '--help')
    assert ret.success
