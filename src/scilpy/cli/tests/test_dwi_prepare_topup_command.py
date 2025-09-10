#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest


@pytest.mark.smoke
def test_help_option(script_runner):
    ret = script_runner.run(['scil_dwi_prepare_topup_command', '--help'])
    assert ret.success
