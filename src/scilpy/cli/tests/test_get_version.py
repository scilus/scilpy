#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest


@pytest.mark.smoke
def test_help_option(script_runner):
    ret = script_runner.run(['scil_get_version', '--help'])
    assert ret.success


@pytest.mark.smoke
def test_execution(script_runner):
    ret = script_runner.run(['scil_get_version'])
    assert ret.success
