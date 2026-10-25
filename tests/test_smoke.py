"""Phase 1 smoke tests: the package imports, the CLI runs, caches stay inside the project."""

import os
from pathlib import Path

import pytest

import pondguard
from pondguard import cli, paths


def test_version_is_set():
    assert pondguard.__version__


def test_cli_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "pondguard" in capsys.readouterr().out


def test_cli_without_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("name", ["HF_HOME", "PHOENIX_WORKING_DIR"])
def test_caches_live_inside_project_folder(name):
    cache = Path(os.environ[name]).resolve()
    assert cache.is_relative_to(paths.PROJECT_ROOT)
    assert not cache.is_relative_to(paths.REPO_ROOT), "caches must sit outside the git repo"


def test_library_telemetry_is_off():
    assert os.environ["ANONYMIZED_TELEMETRY"] == "False"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
