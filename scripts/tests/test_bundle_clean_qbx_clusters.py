#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_bundle_clean_qbx_clusters.py', '--help')
    assert ret.success
