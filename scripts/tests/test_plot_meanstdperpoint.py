#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_plot_meanstdperpoint.py', '--help')
    assert ret.success
