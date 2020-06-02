#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_extract_dwi_shell.py', '--help')
    assert ret.success
