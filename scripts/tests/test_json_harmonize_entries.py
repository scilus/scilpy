#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def test_help_option(script_runner):
    ret = script_runner.run('scil_json_harmonize_entries.py', '--help')
    assert ret.success
