from pathlib import Path
import pytest


def pytest_addoption(parser):
    parser.addoption("--root_path", type=Path, default=None)


@pytest.fixture(scope="session")
def root_path(request):
    root_path = request.config.option.root_path
    if root_path is None:
        pytest.skip()
    return root_path
