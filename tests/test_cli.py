"""Command-line behaviour: short messages for fixable problems, no silent wrong runs."""

import pytest

from pondguard import cli


def test_unknown_scenario_is_a_short_message(capsys):
    assert cli.main(["run", "--scenario", "NOPE-99"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: unknown scenario 'NOPE-99'") and "Traceback" not in err


def test_variant_a_is_refused_not_silently_run_as_b(capsys, monkeypatch):
    monkeypatch.setenv("PONDGUARD_VARIANT", "A")
    assert cli.main(["run", "--scenario", "HEALTHY-01"]) == 1
    assert "not built yet" in capsys.readouterr().err


def test_connection_failures_get_a_fix_hint():
    class ConnectError(Exception):
        pass

    message = cli.explain(ConnectError("refused"))
    assert "lms server start" in message and "LLM_PROVIDER=mock" in message


def test_debug_mode_shows_the_traceback(monkeypatch):
    monkeypatch.setenv("PONDGUARD_DEBUG", "1")
    with pytest.raises(cli.CliError):
        cli.main(["run", "--scenario", "NOPE-99"])
