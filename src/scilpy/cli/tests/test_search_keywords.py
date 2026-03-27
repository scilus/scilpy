#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

pytestmark = pytest.mark.xdist_group("serial")


def test_generate_option(script_runner):
    ret = script_runner.run(['scil_search_keywords', 'test',
                             '--regenerate_help_files',
                             '--processes', '4'])
    assert ret.success


def test_help_option(script_runner):
    ret = script_runner.run(['scil_search_keywords', '--help'])
    assert ret.success


def test_search_category(script_runner):
    ret = script_runner.run(['scil_search_keywords',
                             '--search_category', 'sh'])
    assert 'Available objects:' in ret.stdout


def test_no_synonyms(script_runner):
    ret = script_runner.run(['scil_search_keywords', 'sh', '--no_synonyms'])
    assert ret.success


def test_not_found(script_runner):
    ret = script_runner.run(['scil_search_keywords', 'toto'])
    assert ret.success
    assert 'No results found!' in ret.stdout or 'No results found!' in ret.stderr
