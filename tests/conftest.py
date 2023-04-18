import pytest
def pytest_addoption(parser):
    parser.addoption("--root_path", action="store")
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )

def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.name
    if 'root_path' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("root_path", [option_value])

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skipper = pytest.mark.skip(reason="Only run when --run-slow is given")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skipper)