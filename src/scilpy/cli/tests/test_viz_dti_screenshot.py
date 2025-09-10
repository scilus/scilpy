#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest


@pytest.mark.smoke
def test_help_option(script_runner):
    ret = script_runner.run(['scil_dwi_convert_FDF', '--help'])
    assert ret.success
