"""Stack supervisor — CPU-only: command construction, workspace resolution, probes.

Nothing here launches a process or touches redis (dry-run / pure functions only).
"""
from __future__ import annotations

import socket

from agentbot.settings import REPO_ROOT, load_config
from agentbot.stack.supervisor import StackSupervisor, build_services, port_open


def test_build_services_names_and_commands():
    svcs = {s.name: s for s in build_services(load_config(), headless=True)}
    assert set(svcs) == {"vlm", "gr00t", "sim_session", "dashboard"}
    assert "run_vlm_server.sh" in svcs["vlm"].cmd
    assert "gr00t.eval.run_gr00t_server" in svcs["gr00t"].cmd
    assert "agentbot.vla.sim_session" in svcs["sim_session"].cmd
    assert "uvicorn agentbot.api.app:app" in svcs["dashboard"].cmd


def test_headless_flag_follows_argument():
    cfg = load_config()
    hd = {s.name: s for s in build_services(cfg, headless=True)}
    win = {s.name: s for s in build_services(cfg, headless=False)}
    assert hd["sim_session"].cmd.rstrip().endswith("--headless")
    assert "--headless" not in win["sim_session"].cmd


def test_workspace_paths_resolved_not_hardcoded():
    """cd targets come from workspace.yaml (they exist), never a literal /home/... guess."""
    for s in build_services(load_config(), headless=True):
        assert str(REPO_ROOT) in s.cmd or "conda activate" in s.cmd


def test_port_open_probe():
    with socket.socket() as srv:
        srv.bind(("localhost", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        assert port_open(port) is True
    assert port_open(port) is False


def test_dry_run_prints_without_launching(capsys):
    sup = StackSupervisor()
    services = sup.up(headless=True, dry_run=True)
    out = capsys.readouterr().out
    assert len(services) == 4
    assert out.count("[stack:dry-run]") == 4
    assert "-> pid" not in out          # nothing was actually launched
