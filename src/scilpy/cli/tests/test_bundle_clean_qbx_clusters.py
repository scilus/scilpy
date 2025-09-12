#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

@pytest.mark.smoke
def test_help_option(script_runner):
    ret = script_runner.run(['scil_json_harmonize_entries', '--help'])
    assert ret.success
