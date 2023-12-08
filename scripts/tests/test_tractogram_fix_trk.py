#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_tractogram_fix_trk.py', '--help')
    assert ret.success
