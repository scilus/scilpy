#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_filter_streamlines_by_length.py', '--help')
    assert ret.success
