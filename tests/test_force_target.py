"""force_can_for_color maps an instruction color to the can asset task_done checks.

orange -> red_can (target_object_color 0); green -> blue_can (1). Unknown -> None.
sim_session sets carb /pickplace_env/force_target_object so the env's reset picks that can,
making the commanded color the actual target (terminations.py checks the matching basket).
"""
import importlib.util

# sim_session.py imports isaaclab at module top (env_isaaclab only); load just the helper by
# exec'ing the function's source is brittle, so import lazily and skip if isaaclab is absent.
import pytest

SS = "agentbot.vla.sim_session"


def _load():
    if importlib.util.find_spec("isaaclab") is None:
        pytest.skip("isaaclab not importable in this venv; helper verified on the stack")
    import agentbot.vla.sim_session as m
    return m


def test_force_can_for_color_maps_orange_and_green():
    m = _load()
    assert m.force_can_for_color("orange") == "red_can"
    assert m.force_can_for_color("green") == "blue_can"
    assert m.force_can_for_color("GREEN") == "blue_can"   # case-insensitive
    assert m.force_can_for_color("purple") is None
    assert m.force_can_for_color(None) is None
