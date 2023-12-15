# -*- coding: utf-8 -*-

from glob import glob
from os.path import realpath
from unittest.mock import DEFAULT
import pytest
import warnings


"""
Scilpy Pytest plugin. As of now, this plugin is only used to handle mocking,
but it can be extended to hook into other parts of the testing process.

Mocking interface
-----------------

Writing mocking fixtures is long and tedious. The interface provided by
unittest.mock, and the pytest-mock overhead, is cumbersome. Inasmuch, with
the default pytest framework, it is impossible to share mocks between tests
that are not located within the same module.

This plugin registers early into pytest and thus, can investiguate the modules'
structure and load stuff into the tests namespaces before they are executed.

- It first hooks into pytest_addoption to add a new command line option to
  load mocks for any or all modules in the scilpy package, --mocks.

- It then hooks into pytest_configure to load the mocks activated by the user
  into the test session. Note that this way, all mocks associated with a module
  gets injected into pytest's namespace, which might not be granular enough for
  some use cases (see below).

To make mock creation and collection in test instances, and to allow for a more
granular selection of mocks from mocking modules, this plugin provides two
fixtures:

- mock_creator : the mock_creator fixture exposes the bases interface of 
                 unittest.mock patchers, but with a more convenient syntax. It
                 is able to patch mutliple attributes at once, and can be
                 configured to create the patched object if it does not exist.

- mock_collector : the mock_collector fixture is a helper function that is used
                   to collect specific mocks from mocking modules. It is also
                   used to modify the namespace into which the mocked objects
                   get patched. This is required for some mocks to be used with
                   scripts, when their import is relative (e.g. from . import).

Creating a new mock is done using the mock_creator fixture. All mocks must be
placed inside the scilpy library, in the tests directories of their respective
modules, in fixtures/mocks.py. This is own they get discovered by the plugin.

- A mock fixture must have a relevant name (e.g. amico_evaluator patches several
  parts of the amico.Evaluation class). Its return value is the result of
  calling the mock_creator fixture.

- The mock_creator fixture does not need to be imported, it is provided
  automatically by the pytest framework. Simply add mock_creator as a parameter
  to the mock fixture function.

Using mocks in tests is done using the mock_collector fixture. Like the
mock_creator, it is provided automatically by the pytest framework. Simply
add mock_collector as a parameter to the test function that requires mocking. To
use the mocks, call the mock_collector fixture with the list of mock names to
use. Additionally, the mock_collector fixture can be used to modify the
namespace into which the mocks are injected, by providing a patch_path argument
as a second parameter. The returned dictionary indexes loaded mocks by their
name and can be used to assert their usage throughout the test case.
"""


AUTOMOCK = DEFAULT

# Load mock modules from all library tests

MOCK_MODULES = list(_module.replace("/", ".").replace(".py", "")
                    for _module in glob("**/tests/fixtures/mocks.py",
                                        root_dir=realpath('.'),
                                        recursive=True))

MOCK_PACKAGES = list(m.split(".")[-4] for m in MOCK_MODULES)


# Helper function to select mocks

def _get_active_mocks(include_list=None, exclude_list=None):
    """
    Returns a list of all packages with active mocks in the current session

    Parameters
    ----------
    include_list: iterable or None
        List of scilpy packages to consider

    exclude_list: iterable or None
        List of scilpy packages to exclude from consideration

    Returns
    -------
    list: list of packages with active mocks
    """

    def _active_names():
        def _exclude(_l):
            if exclude_list is None:
                return _l
            return filter(lambda _i: _i not in exclude_list, _l)

        if include_list is None or len(include_list) == 0:
            return []

        if "all" in include_list:
            return _exclude(MOCK_PACKAGES)

        return _exclude([_m for _m in include_list if _m in MOCK_PACKAGES])

    return list(map(lambda _m: _m[1],
                    filter(lambda _m: _m[0] in _active_names(),
                           zip(MOCK_PACKAGES, MOCK_MODULES))))


# Create hooks and fixtures to handle mocking from pytest command line

def pytest_addoption(parser):
    parser.addoption(
        "--mocks",
        nargs='+',
        choices=["all"] + MOCK_PACKAGES,
        help="Load mocks for scilpy packages to accelerate"
             "tests and prevent testing external dependencies")


def pytest_configure(config):
    _toggle_mocks = config.getoption("--mocks")
    for _mock_mod in _get_active_mocks(_toggle_mocks):
        config.pluginmanager.import_plugin(_mock_mod)


@pytest.fixture
def mock_collector(request):
    """
    Pytest fixture to collect a specific set of mocks for a test case
    """
    def _collector(mock_names, patch_path=None):
        try:
            return {_name: request.getfixturevalue(_name)(patch_path)
                    for _name in mock_names}
        except pytest.FixtureLookupError:
            warnings.warn(f"Some fixtures in {mock_names} cannot be found.")
            return None
    return _collector


@pytest.fixture
def mock_creator(mocker):
    """
    Pytest fixture to create a namespace patchable mock
    """
    def _mocker(base_module, object_name, side_effect=None,
                mock_attributes=None):

        def _patcher(module_name=None):
            _base = base_module if module_name is None else module_name

            if mock_attributes is not None:
                return mocker.patch.multiple("{}.{}".format(_base, object_name),
                                             **{a: AUTOMOCK
                                                for a in mock_attributes})

            return mocker.patch("{}.{}".format(_base, object_name),
                                side_effect=side_effect, create=True)

        return _patcher

    return _mocker
