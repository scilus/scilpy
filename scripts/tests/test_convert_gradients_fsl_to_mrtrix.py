#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_convert_gradients_fsl_to_mrtrix.py', '--help')
    assert ret.success
