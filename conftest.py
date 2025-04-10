import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--ml", action="store_true", default=False,
        help="Run machine learning tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "ML: mark test that refer to machine learning")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--ml"):
        # --ml given in cli: do not skip ml tests
        return
    skip_ml = pytest.mark.skip(reason="need --ml option to run")
    for item in items:
        if "ml" in item.keywords:
            item.add_marker(skip_ml)
