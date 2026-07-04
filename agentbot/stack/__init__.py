"""Cross-repo sim-stack supervisor — agentbot wakes and monitors the whole stack.

Absorbs the former ``scripts/stack/{start,stop}_stack.sh``:

    python -m agentbot.stack up [--headless|--windowed] [--dry-run]
    python -m agentbot.stack status
    python -m agentbot.stack down
    python -m agentbot.stack logs
"""
from agentbot.stack.supervisor import Service, StackSupervisor, build_services, port_open

__all__ = ["Service", "StackSupervisor", "build_services", "port_open"]
