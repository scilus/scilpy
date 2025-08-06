#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run(['scil_get_version', '--help'])
    assert ret.success


def test_execution(script_runner):
    ret = script_runner.run(['scil_get_version'])
    assert ret.success
