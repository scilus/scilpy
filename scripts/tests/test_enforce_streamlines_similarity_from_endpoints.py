#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_add_tracking_mask_to_pft_maps.py', '--help')
    assert ret.success
