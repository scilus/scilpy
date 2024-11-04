#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_mrds_select_number_of_tensors.py', '--help')
    assert ret.success
