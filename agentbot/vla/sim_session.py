"""Persistent IsaacLab session (block 5, Phase 2).

Keeps the IsaacLab app OPEN and executes skills one at a time off the Redis job
queue (``agentbot:vla:tasks``), publishing telemetry + a terminal result per skill
to the event stream — so the app launches once and survives across many skills
(unlike the Phase-1 per-episode subprocess).

Run in env_isaaclab AFTER the GR00T server is up:
    conda activate env_isaaclab && python -m agentbot.vla.sim_session
No-Redis verification of the persistent loop (still needs the GR00T server + GPU):
    python -m agentbot.vla.sim_session --selftest

This file refactors the OUTER loop of scripts/eval/gr00t_infer_agent.py into a
command-driven loop; the per-chunk obs->action->step inner loop is lifted verbatim.
AppLauncher MUST be constructed before importing gym/isaaclab (hard requirement).
"""
# --------------------------------------------------------------------------- #
#  1. Launch the simulator FIRST (before any gym / isaaclab import).
# --------------------------------------------------------------------------- #
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="AgentBot persistent IsaacLab session.")
parser.add_argument("--task", default=None, help="task gym id (default: config vla.default_task)")
parser.add_argument("--max-skills", type=int, default=0, help="exit after N skills (0 = forever)")
parser.add_argument("--selftest", action="store_true", help="run 2 hardcoded skills (no Redis) and exit")
parser.add_argument("--openarm_hand_type", default="linkerhand_o6")
parser.add_argument("--pov", default="head", choices=["head", "wrist_R", "wrist_L"])
parser.add_argument("--max_steps", type=int, default=600, help="max env steps per skill before truncating")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import pinocchio  # noqa: F401  (force IsaacLab's pinocchio before AppLauncher)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
#  2. Everything else (now safe to import gym / isaaclab / torch).
# --------------------------------------------------------------------------- #
import sys
import time
from typing import cast

import carb
import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg

from agentbot.contracts.events import Event, EventType
from agentbot.contracts.vla import VlaTaskRequest, VlaTaskResult, VlaTaskStatus, VlaTelemetry
from agentbot.settings import REPO_ROOT, load_config
from agentbot.vla.frame_encode import frame_to_datauri
from agentbot.vla.isaac_runner import _telemetry_event, backend_spec
from agentbot.vla.worker import _result_event, _status_event

# eval utils (joint mapping / filter / GR00T client) — same proven code as the eval harness.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))
from utils.filter import LowPassFilter           # noqa: E402
from utils.gr00t_client_adapter import Gr00tClientAdapter  # noqa: E402
from utils.joint_mapper import JointMapper       # noqa: E402

CAMERA_OBS = {"head": "rgb_image", "wrist_L": "wrist_L_image", "wrist_R": "wrist_R_image"}
STABILIZATION_STEPS = 5

# Per-skill robot type + joint-space action mode (same carb settings as the eval agent).
_carb = carb.settings.get_settings()
_carb.set_bool("/current_env/use_joint_space", True)
ROBOT_TYPE = "openarm_" + args_cli.openarm_hand_type
_carb.set_string("/current_env/robot_type", ROBOT_TYPE)


def import_task_done(task_name: str):
    """Task-specific success predicate (mirrors gr00t_infer_agent.py's conditional import)."""
    base = "isaaclab_tasks.manager_based.manipulation.playground_openarm.dexhand_bimanual.task_scenes"
    import importlib
    scene = None
    if "Can-Sorting" in task_name:
        scene = "can_sorting"
    elif "Pour-Water" in task_name:
        scene = "pour_water"
    elif "Cube-Stack" in task_name:
        scene = "cube_stack"
    elif "Cabinet-Pour" in task_name:
        scene = "cabinet_pour"
    if scene is None:
        return lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    mod = importlib.import_module(f"{base}.{scene}.mdp.terminations")
    return mod.task_done


def run_stabilization(env, idle_actions):
    for _ in range(STABILIZATION_STEPS):
        obs, _, _, _, _ = env.step(idle_actions)
    return obs


FORCE_TARGET_SETTING = "/pickplace_env/force_target_object"
COLOR_TO_CAN = {"orange": "red_can", "green": "blue_can"}


def force_can_for_color(color):
    """Instruction color -> the can asset task_done checks ('orange'->red_can, 'green'->blue_can),
    or None for unknown. Mirrors target_object_color {0: red_can/orange, 1: blue_can/green}."""
    return COLOR_TO_CAN.get((color or "").lower())


def set_force_target(color) -> None:
    """Pin the env's next reset target to *color*'s can (so the commanded color becomes the
    actual target). Unknown color clears the pin (-> random)."""
    _carb.set_string(FORCE_TARGET_SETTING, force_can_for_color(color) or "")


def clear_force_target() -> None:
    """Release the pin so the env re-randomizes its target on reset."""
    _carb.set_string(FORCE_TARGET_SETTING, "")


def _scene_target_color(obs):
    """The env's current randomized target plate color ('orange'|'green'), or None.

    Single source of truth for both the instruction sync (run_skill) and the Monitor
    state the Brain reads to resolve a bare 'sort can'. {0: orange, 1: green} mirrors
    gr00t_infer_agent.py's TARGET_COLOR_ID_TO_NAME."""
    try:
        cid = int(obs["scene_obs"]["target_object_color"].cpu().numpy().reshape(-1)[0])
        return {0: "orange", 1: "green"}.get(cid)
    except (KeyError, TypeError, IndexError, AttributeError, ValueError):
        return None


def publish_scene(obs, publish, task_name: str) -> None:
    """Publish a fresh head-camera frame + the env's target color to the Monitor.

    Lets the dashboard show the live scene while idle, and lets the Brain turn a bare
    'sort can' into 'place the can on the <color> plate' from state['environment']."""
    try:
        img = obs["robot_obs"][CAMERA_OBS["head"]].cpu().numpy().astype(np.uint8)
        publish(Event(type=EventType.CAMERA, source="vla.sim_session",
                      payload={"frame": frame_to_datauri(img)}))
    except (KeyError, TypeError, AttributeError):
        pass
    color = _scene_target_color(obs)
    if color:
        publish(Event(type=EventType.ENVIRONMENT, source="vla.sim_session",
                      payload={"target_color": color, "task": task_name}))


def build_env(task_name: str):
    """Create env + JointMapper + idle action + task_done for *task_name* (keeps app alive)."""
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=1,
                            use_fabric=not args_cli.disable_fabric)
    env_cfg.terminations.success = None            # success judged by task_done()
    env = cast(ManagerBasedRLEnv, gym.make(task_name, cfg=env_cfg).unwrapped)
    robot = env.scene.articulations["robot"]
    mapper = JointMapper(env_cfg=env_cfg, robot_articulation=robot)
    joint_names = env_cfg.actions.arm_action_cfg.joint_names
    defaults = env_cfg.scene.robot.init_state.joint_pos
    idle_np = np.zeros(len(joint_names), dtype=np.float32)
    for i, n in enumerate(joint_names):
        idle_np[i] = defaults.get(n, 0.0)
    idle = torch.tensor(idle_np, dtype=torch.float32, device=env.device).unsqueeze(0)
    task_done = import_task_done(task_name)
    return env, mapper, idle, task_done


def run_skill(env, mapper, idle, task_done, client, req: VlaTaskRequest, publish, abort=None) -> VlaTaskResult:
    """Run ONE episode for *req* in the live env; publish telemetry + return the result.

    Inner obs->action->step loop is lifted verbatim from gr00t_infer_agent.py (~282-345)."""
    task_description = [req.instruction]
    cameras = [args_cli.pov]
    cam_keys = [CAMERA_OBS[c] for c in cameras]
    filt = LowPassFilter(alpha=0.3)
    success = terminated = truncated = False
    aborted = False
    steps = 0
    t0 = time.time()

    # The whole episode (reset + stabilize + step) runs under ONE inference_mode, like
    # gr00t_infer_agent.py — otherwise env.reset() between skills hits "inplace update to
    # inference tensor outside InferenceMode" on tensors the previous skill's steps created.
    with torch.inference_mode():
        if abort is not None:
            abort.clear()               # stale abort from a previous skill must not kill this one
        # Honor the commanded color: pin the env's target to this color's can BEFORE reset, so
        # the randomized target == the command (task_done then checks the right basket). The
        # color-sync below becomes a no-op when they already match; it stays as a safety net.
        set_force_target(req.params.get("target_color"))
        obs, _ = env.reset()
        obs = run_stabilization(env, idle)
        publish(_status_event(req, "running", task_name=req.task_name, instruction=req.instruction))

        # The Can-Sorting env RANDOMIZES the target plate on every reset and judges success
        # (task_done) against THAT randomized basket. So we must drive the policy with the
        # env's actual target, not the Brain's commanded color — otherwise the policy chases
        # the wrong plate, task_done is never True, and the orchestrator retries/replans
        # forever (the "action repeats / always failed though it completed" symptom).
        # This mirrors gr00t_infer_agent.py --multitask (TARGET_COLOR_ID_TO_NAME {0:orange,1:green}).
        # set_force_target() above already pinned the env target to the commanded color (when one was
        # given), so env target == command and this sync just confirms it. When no color was forced
        # (e.g. after ↺ Reset env → random target), it reads the actual target and drives the policy
        # to it. Either way the instruction matches the basket task_done checks.
        color = _scene_target_color(obs)
        if color:
            publish(Event(type=EventType.ENVIRONMENT, source="vla.sim_session",
                          payload={"target_color": color, "task": req.task_name}))
            synced = [f"place the can on the {color} plate"]
            if synced != task_description:
                print(f"[sim_session] env target={color}; syncing instruction "
                      f"(commanded {task_description[0]!r})", flush=True)
            task_description = synced

        while steps < args_cli.max_steps:
            joint_pos = obs["robot_obs"]["robot_joint_pos"].cpu().numpy().astype(np.float64)[0]
            cam_imgs = {c: obs["robot_obs"][k].cpu().numpy().astype(np.uint8) for c, k in zip(cameras, cam_keys)}
            if "head" in cam_imgs:               # publish the live frame so the Brain can see it
                publish(Event(type=EventType.CAMERA, source="vla.sim_session",
                              payload={"frame": frame_to_datauri(cam_imgs["head"])}))
            gr00t_obs = {"annotation.human.task_description": task_description,
                         **mapper.map_isaac_obs_to_gr00t_state(joint_pos)}
            for c, img in cam_imgs.items():
                gr00t_obs["video.camera" if c == "head" else f"video.camera_{c}"] = img

            t_inf = time.time()
            action = client.get_action(gr00t_obs)
            lat = time.time() - t_inf

            env_action = mapper.map_gr00t_action_to_isaac_action(action)
            seqs = torch.tensor(env_action, dtype=torch.float32, device=env.device).unsqueeze(1)
            if req.options.get("filter", True):
                seqs = filt.filter(seqs)

            for a in seqs:
                obs, _, terminated, truncated, _ = env.step(a)
                success = bool(task_done(env).cpu().numpy()[0])
                steps += 1
                if terminated or truncated or success:
                    break

            publish(_telemetry_event(req, VlaTelemetry(task_id=req.task_id, step=steps,
                                                       inference_latency_s=lat, success=success)))
            if abort is not None and abort.tripped(req.task_id):
                aborted = True
                print(f"[sim_session] abort requested; ending episode at step {steps}", flush=True)
                break
            if terminated or truncated or success:
                break

    if aborted:
        status = VlaTaskStatus.ABORTED
    elif success:
        status = VlaTaskStatus.SUCCEEDED
    else:
        status = VlaTaskStatus.FAILED
    result = VlaTaskResult(
        task_id=req.task_id,
        status=status,
        episodes=1, success=1 if success else 0, success_rate=1.0 if success else 0.0,
        steps=steps, duration_s=round(time.time() - t0, 2),
    )
    publish(_result_event(req, result))
    return result


def _make_client(cfg):
    spec = backend_spec(cfg.vla.gr00t_ver)
    # Put the matching gr00t checkout (e.g. Isaac-GR00T_n1d7 for N1.7) on sys.path BEFORE
    # the adapter imports `gr00t`, so the client's ZMQ wire format matches the server's
    # (same as run_eval.py's client_pythonpath). Without this the adapter picks up the
    # N1.6 Isaac-GR00T and the N1.7 server rejects the obs.
    cpp = spec.get("client_pythonpath")
    if cpp:
        sys.path.insert(0, str(REPO_ROOT / cpp))
    return Gr00tClientAdapter(version=cfg.vla.gr00t_ver,
                              host=spec.get("host", "localhost"), port=int(spec.get("port", 5555)))


def _selftest(cfg) -> None:
    """Run two hardcoded can-sorting skills with the app open across both (no Redis)."""
    task = args_cli.task or cfg.vla.default_task
    client = _make_client(cfg)
    env, mapper, idle, task_done = build_env(task)

    def publish(ev: Event) -> None:
        p = ev.payload
        if p.get("step"):
            print(f"  [tel] step={p['step']} lat={p.get('inference_latency_s')} success={p.get('success')}", flush=True)
        else:
            print(f"  [evt] {ev.type.value}: {p.get('status') or p.get('kind')}", flush=True)

    reqs = [
        VlaTaskRequest(task_name=task, instruction="place the can on the orange plate",
                       checkpoint=cfg.vla.resolve_checkpoint(None), gr00t_ver=cfg.vla.gr00t_ver,
                       params={"target_color": "orange"}),
        VlaTaskRequest(task_name=task, instruction="place the can on the green plate",
                       checkpoint=cfg.vla.resolve_checkpoint(None), gr00t_ver=cfg.vla.gr00t_ver,
                       params={"target_color": "green"}),
    ]
    for i, req in enumerate(reqs, 1):
        print(f"[selftest] skill {i}/{len(reqs)}: {req.instruction}", flush=True)
        res = run_skill(env, mapper, idle, task_done, client, req, publish)
        print(f"[selftest] -> {res.status.value} success={res.success} steps={res.steps} ({res.duration_s}s)", flush=True)
    env.close()
    print("[selftest] DONE — the IsaacLab app stayed open across both skills.", flush=True)


def _serve(cfg) -> None:
    """Persistent Redis loop: BRPOP a VlaTaskRequest, run it, publish, repeat."""
    import redis
    r = redis.from_url(cfg.redis.url)
    tasks_key = cfg.redis.tasks_queue
    stream = cfg.redis.events_stream

    def publish(ev: Event) -> None:
        r.xadd(stream, {"json": ev.model_dump_json()})

    client = _make_client(cfg)
    from agentbot.monitor.abort import RedisAbortFlag
    abort = RedisAbortFlag(cfg.redis.url, cfg.redis.abort_key)
    abort.clear()                         # startup: clear any stale abort
    current_task = args_cli.task or cfg.vla.default_task
    env, mapper, idle, task_done = build_env(current_task)

    # Reset once at startup so the dashboard shows a fresh scene + target color immediately
    # (and so the idle keepalive steps an initialized env, not an un-reset one).
    with torch.inference_mode():
        clear_force_target()           # startup scene is freshly random
        obs, _ = env.reset()
        obs = run_stabilization(env, idle)
    publish_scene(obs, publish, current_task)
    last_scene_pub = time.time()
    print(f"[sim_session] ready; task={current_task}; polling {tasks_key}", flush=True)

    n = 0
    while simulation_app.is_running():
        # Non-blocking RPOP (FIFO with the producers' LPUSH). We must NOT block the main
        # thread on Redis — Omniverse needs simulation_app.update() pumped during idle or
        # the app stalls. So poll, and pump+sleep when the queue is empty.
        raw = r.rpop(tasks_key)
        if raw is None:
            # Keep the kit app alive while idle (stepping idle actions, not just
            # simulation_app.update()), and re-publish the live scene (frame + target color)
            # ~1/s so the Brain always has a current view to resolve a bare "sort can".
            with torch.inference_mode():
                obs_idle, _, _, _, _ = env.step(idle)
            now = time.time()
            if now - last_scene_pub > 1.0:
                publish_scene(obs_idle, publish, current_task)
                last_scene_pub = now
            time.sleep(0.01)
            continue
        req = VlaTaskRequest.model_validate_json(raw)
        if req.task_name and req.task_name != current_task:
            env.close()
            current_task = req.task_name
            env, mapper, idle, task_done = build_env(current_task)
        if req.params.get("control") == "reset":
            # Control op (from POST /v1/control/reset): re-randomize the scene + re-publish;
            # no policy episode is run.
            print(f"[sim_session] reset scene (task={current_task})", flush=True)
            with torch.inference_mode():
                clear_force_target()   # ↺ Reset env => fresh random target
                obs, _ = env.reset()
                obs = run_stabilization(env, idle)
            publish_scene(obs, publish, current_task)
            last_scene_pub = time.time()
            continue
        run_skill(env, mapper, idle, task_done, client, req, publish, abort=abort)
        abort.clear()                     # done/aborted -> release so the next skill runs
        last_scene_pub = time.time()
        n += 1
        if args_cli.max_skills and n >= args_cli.max_skills:
            break
    env.close()


def main() -> None:
    cfg = load_config()
    if args_cli.selftest:
        _selftest(cfg)
    else:
        _serve(cfg)


if __name__ == "__main__":
    import signal

    def _term(*_):                       # turn SIGTERM into a clean shutdown
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    try:
        main()
    except KeyboardInterrupt:
        print("[sim_session] shutting down", flush=True)
    finally:
        # Always shut Omniverse down — a crash/term mid-skill must not leave a hung app
        # holding the GPU. (SIGTERM now reaches here, so `kill` works without -9.)
        simulation_app.close()
