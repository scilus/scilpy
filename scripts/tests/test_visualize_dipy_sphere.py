#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from unittest.mock import patch


def test_help_option(script_runner):
    ret = script_runner.run('scil_visualize_dipy_sphere.py', '--help')
    assert ret.success


@patch("matplotlib.pyplot.show")
def test_execution_connectivity(script_runner):
    ret = script_runner.run('scil_visualize_dipy_sphere.py',
                            'symmetric724')
    assert ret.success
