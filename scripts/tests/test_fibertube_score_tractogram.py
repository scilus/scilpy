#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_fibertube_score_tractogram.py', '--help')
    assert ret.success
