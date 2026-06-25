#!/usr/bin/env bash
# Supervisor for the persistent IsaacLab sim_session.
#
# Headless Omniverse Kit self-quits ("quick framework shutdown") after ~1 min of
# idle even while polling — an upstream isaacsim behaviour we couldn't disable from
# Python. This supervisor relaunches sim_session whenever it exits, so the agentbot
# redis job queue is always drained: a command buffers in redis and runs as soon as
# a session is up (and an in-progress episode never idles, so it always completes).
#
# Run inside env_isaaclab:
#   conda activate env_isaaclab && bash agentbot/run_sim_session.sh
set -u
export OMNI_KIT_ACCEPT_EULA=YES
ISAACLAB=${ISAACLAB:-/home/asus/Gits/IsaacLab-GR00T/IsaacLab}
cd "$ISAACLAB"
n=0
while true; do
  n=$((n + 1))
  echo "[supervisor] launch #$n sim_session $(date +%T)"
  python -m agentbot.vla.sim_session --headless
  echo "[supervisor] sim_session exited (rc=$?); relaunch in 3s $(date +%T)"
  sleep 3
done
