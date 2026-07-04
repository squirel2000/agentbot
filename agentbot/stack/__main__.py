"""CLI: ``python -m agentbot.stack up|status|down|logs``."""
from __future__ import annotations

import argparse
import json

from agentbot.stack.supervisor import StackSupervisor


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agentbot.stack", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_up = sub.add_parser("up", help="launch redis + VLM + GR00T + sim_session + dashboard")
    g = p_up.add_mutually_exclusive_group()
    g.add_argument("--headless", dest="headless", action="store_true", default=None,
                   help="no sim window (idle Kit self-quits)")
    g.add_argument("--windowed", dest="headless", action="store_false",
                   help="show the sim window (default per config)")
    p_up.add_argument("--dry-run", action="store_true",
                      help="print the exact commands without launching anything")

    sub.add_parser("status", help="pid + port health of every service (JSON)")
    sub.add_parser("down", help="stop everything and clear redis queues")
    sub.add_parser("logs", help="list per-service log files")

    args = ap.parse_args(argv)
    sup = StackSupervisor()
    if args.command == "up":
        sup.up(headless=args.headless, dry_run=args.dry_run)
    elif args.command == "status":
        print(json.dumps(sup.status(), indent=2))
    elif args.command == "down":
        sup.down()
    elif args.command == "logs":
        sup.logs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
