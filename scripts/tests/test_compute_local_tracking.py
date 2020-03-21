#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_local_tracking.py', '--help')
    assert ret.success
