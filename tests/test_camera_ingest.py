"""C0 (CPU part): CAMERA event -> state["camera"] via Ingestor."""
from agentbot.contracts.events import Event, EventType
from agentbot.monitor.ingest import Ingestor
from agentbot.monitor.state_store import InMemStateStore


def test_camera_event_lands_in_state():
    st = InMemStateStore()
    ing = Ingestor(bus=None, state_store=st)
    ing.handle(Event(type=EventType.CAMERA, source="vla.sim_session",
                     payload={"frame": "data:image/jpeg;base64,QQ==", "ts": 1.0}))
    assert st.get_state("camera")["frame"].startswith("data:image/jpeg")


def test_camera_event_full_payload_stored():
    """The entire payload dict (frame + ts) is stored under 'camera'."""
    st = InMemStateStore()
    ing = Ingestor(bus=None, state_store=st)
    ing.handle(Event(type=EventType.CAMERA, source="vla.sim_session",
                     payload={"frame": "data:image/png;base64,aGk=", "ts": 42.5}))
    cam = st.get_state("camera")
    assert cam["ts"] == 42.5
    assert "frame" in cam


def test_non_camera_event_does_not_touch_camera_key():
    """Unrelated events must not overwrite or create the camera key."""
    st = InMemStateStore()
    ing = Ingestor(bus=None, state_store=st)
    ing.handle(Event(type=EventType.ROBOT_STATE, source="robot",
                     payload={"mode": "idle", "detail": ""}))
    assert st.get_state("camera") == {}
