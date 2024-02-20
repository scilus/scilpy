#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_search_keywords.py', '--help')
    assert ret.success


def test_no_verbose(script_runner):
    ret = script_runner.run('scil_search_keywords.py', 'mti')
    assert ret.success


def test_verbose_option(script_runner):
    ret = script_runner.run('scil_search_keywords.py', 'mti', '-v')
    assert ret.success


def test_not_find(script_runner):
    ret = script_runner.run('scil_search_keywords.py', 'toto')
    assert ret.success
