import pathlib


class OpenmapT1TestCase(object):
    PROJECT_ROOT = pathlib.Path(__file__).parents[2].resolve()
    MODULE_ROOT = PROJECT_ROOT / "src" / "openmap_t1"
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"
