"""Monitor (block 3): the in-process event bus, state store, job queue, and the
safety watchdog that closes the immediate-stop path."""
import asyncio

from agentbot.monitor.event_bus import InProcEventBus
from agentbot.monitor.state_store import InMemStateStore
from agentbot.monitor.job_queue import InMemJobQueue
from agentbot.monitor.safety import SafetyWatchdog
from agentbot.monitor.ingest import Ingestor
from agentbot.contracts.events import Event, EventType, SafetyEvent, RobotState
from agentbot.contracts.vla import VlaTaskRequest


async def test_inproc_pub_sub_filters_by_type():
    bus = InProcEventBus()
    got: list[Event] = []

    async def reader():
        async for ev in bus.subscribe([EventType.SAFETY]):
            got.append(ev)
            break

    t = asyncio.create_task(reader())
    await asyncio.sleep(0)  # let the subscriber register
    await bus.publish(Event(type=EventType.SYSTEM_STATUS, source="x"))   # filtered out
    await bus.publish(Event(type=EventType.SAFETY, source="x",
                            payload=SafetyEvent(kind="collision", immediate=True).model_dump()))
    await asyncio.wait_for(t, timeout=1)
    assert len(got) == 1 and got[0].type == EventType.SAFETY


async def test_safety_watchdog_trips_on_immediate():
    bus = InProcEventBus()
    tripped: list[Event] = []
    wd = SafetyWatchdog(bus, on_stop=lambda ev: tripped.append(ev))
    task = asyncio.create_task(wd.run())
    await asyncio.sleep(0)
    await bus.publish(Event(type=EventType.SAFETY, source="env",
                            payload=SafetyEvent(kind="emergency", immediate=True).model_dump()))
    await asyncio.sleep(0.02)
    task.cancel()
    assert tripped and tripped[0].type == EventType.SAFETY


async def test_safety_watchdog_ignores_non_immediate():
    bus = InProcEventBus()
    tripped: list[Event] = []
    wd = SafetyWatchdog(bus, on_stop=lambda ev: tripped.append(ev))
    task = asyncio.create_task(wd.run())
    await asyncio.sleep(0)
    await bus.publish(Event(type=EventType.SAFETY, source="env",
                            payload=SafetyEvent(kind="violation", immediate=False).model_dump()))
    await asyncio.sleep(0.02)
    task.cancel()
    assert tripped == []


def test_state_store_snapshot():
    st = InMemStateStore()
    st.set_state("robot_state", RobotState(mode="running", detail="arm moving").model_dump())
    assert st.get_state("robot_state")["mode"] == "running"
    assert st.snapshot()["robot_state"]["mode"] == "running"


def test_job_queue_roundtrip():
    q = InMemJobQueue()
    req = VlaTaskRequest(task_name="t", instruction="i", checkpoint="c")
    q.put(req)
    got = q.get(timeout=0.1)
    assert got is not None and got.task_id == req.task_id
    assert q.get(timeout=0.01) is None   # empty now


def test_ingestor_updates_live_state_and_vla_result():
    st = InMemStateStore()
    ing = Ingestor(InProcEventBus(), st)
    ing.handle(Event(type=EventType.ROBOT_STATE, source="sim",
                     payload=RobotState(mode="running", detail="arm moving").model_dump()))
    assert st.snapshot()["robot_state"]["mode"] == "running"
    # VLA telemetry then terminal result merge under vla:<task_id>
    ing.handle(Event(type=EventType.VLA_TELEMETRY, source="vla.worker", task_id="task_1",
                     payload={"status": "running"}))
    ing.handle(Event(type=EventType.VLA_TELEMETRY, source="vla.worker", task_id="task_1",
                     payload={"kind": "result", "status": "succeeded", "success_rate": 1.0}))
    vla = st.get_state("vla:task_1")
    assert vla["status"] == "succeeded" and vla["success_rate"] == 1.0
