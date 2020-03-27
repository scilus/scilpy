#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_summarize_bundles_measures.py', '--help')
    assert ret.success
