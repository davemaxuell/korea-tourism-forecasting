from __future__ import annotations

import sys

import pytest

from korea_tourism import __main__, data, evaluation, fetch


@pytest.mark.parametrize("command", ["build", "evaluate", "reproduce", "fetch"])
def test_cli_routes_explicit_workspace_and_seed(tmp_path, monkeypatch, command):
    calls = []
    monkeypatch.setattr(data, "build_feature_table", lambda root: calls.append(("build", root)))
    monkeypatch.setattr(
        evaluation, "run_evaluation", lambda root, seed: calls.append(("evaluate", root, seed))
    )
    monkeypatch.setattr(
        fetch,
        "fetch_exchange_rates",
        lambda root, start, end: calls.append(("fetch", root, start, end)),
    )
    argv = ["korea-tourism", command, "--root", str(tmp_path)]
    if command in ("evaluate", "reproduce"):
        argv += ["--seed", "7"]
    monkeypatch.setattr(sys, "argv", argv)
    __main__.main()
    assert all(call[1] == tmp_path.resolve() for call in calls)
    expected = ["build", "evaluate"] if command == "reproduce" else [command]
    assert [call[0] for call in calls] == expected
    if command in ("evaluate", "reproduce"):
        assert calls[-1][2] == 7


def test_missing_command_is_usage_error(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["korea-tourism"])
    with pytest.raises(SystemExit) as error:
        __main__.main()
    assert error.value.code == 2
