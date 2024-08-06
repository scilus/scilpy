#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_help_option(script_runner):
    ret = script_runner.run('scil_search_keywords.py', '--help')
    assert ret.success


def test_search_category(script_runner):
    ret = script_runner.run('scil_search_keywords.py', '--search_category', 'sh')
    assert 'Available objects:' in ret.stdout


def test_no_synonyms(script_runner):
    ret = script_runner.run('scil_search_keywords.py', 'sh', '--no_synonyms')
    assert ret.success


def test_not_found(script_runner):
    ret = script_runner.run('scil_search_keywords.py', 'toto')
    assert ret.success
    assert 'No results found!' in ret.stdout or 'No results found!' in ret.stderr
