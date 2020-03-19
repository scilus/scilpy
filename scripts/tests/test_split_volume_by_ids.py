#!/usr/bin/env python
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_split_volume_by_ids.py', '--help')
    assert ret.success
