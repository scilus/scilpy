import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--apply-mocks",
        action="store_true",
        help="Apply mocks to accelerate tests and "
             "prevent testing external dependencies"
    )


@pytest.fixture(scope="session")
def apply_mocks(request):
    return request.config.getoption("--apply-mocks")