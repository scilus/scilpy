#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_local_tracking_gpu.py', '--help')
    assert ret.success
