#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_generate_priors_from_bundle.py', '--help')
    assert ret.success
