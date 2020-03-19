#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_summarize_bundles_similarity_measures.py', '--help')
    assert ret.success
