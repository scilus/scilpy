#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_apply_warp_to_tractogram.py', '--help')
    assert ret.success
