import pytest


@pytest.fixture(scope="session")
def apply_mocks(request):
    return request.config.getoption("--apply-mocks")
