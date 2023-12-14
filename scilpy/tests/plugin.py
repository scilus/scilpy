# -*- coding: utf-8 -*-

from glob import glob
from os.path import realpath
import pytest


# Load mock modules from all library tests

MOCK_MODULES = list(_module.replace("/", ".").replace(".py", "")
                    for _module in glob("**/tests/fixtures/mocks.py",
                                        root_dir=realpath('.'),
                                        recursive=True))

MOCK_NAMES = list(m.split(".")[-1] for m in MOCK_MODULES)

LOADED_MOCK_MODULES = []


# Helper function to select mocks

def _get_active_mocks(include_list=None, exclude_list=None):
    """
    Returns a list of all active mocks in the current session

    Parameters
    ----------
    include_list: iterable or None
        List of mocks to consider

    exclude_list: iterable or None
        List of mocks to exclude from consideration

    Returns
    -------
    list: list of modules containing mocks
    """

    def _active_names():
        def _exclude(_l):
            if exclude_list is None:
                return _l
            return filter(lambda _i: _i not in exclude_list, _l)

        if include_list is None or len(include_list) == 0:
            return []

        if "all" in include_list:
            return _exclude(MOCK_NAMES)

        return _exclude([_m for _m in include_list if _m in MOCK_NAMES])

    return list(map(lambda _m: _m[1],
                    filter(lambda _m: _m[0] in _active_names(),
                           zip(MOCK_NAMES, MOCK_MODULES))))


# Create hooks and fixtures to handle mocking from pytest command line

def pytest_addoption(parser):
    parser.addoption(
        "--mocks",
        nargs='+',
        choices=["all"] + MOCK_NAMES,
        help="Apply mocks to accelerate tests and "
             "prevent testing external dependencies")


def pytest_configure(config):
    _toggle_mocks = config.getoption("--mocks")
    for _mock_mod in _get_active_mocks(_toggle_mocks):
        config.pluginmanager.import_plugin(_mock_mod)
        LOADED_MOCK_MODULES.append(_mock_mod)


@pytest.fixture
def mock_collector(request):
    def _collector(mock_name):
        _fixture = request(mock_name)
        print(_fixture)
        return _fixture.getfixturevalue()
    return _collector


@pytest.fixture
def mock_creator(module_name, object_name, mocker,
                 side_effect=None):

    return mocker.patch("{}.{}".format(module_name, object_name),
                        side_effect=side_effect, create=True)
