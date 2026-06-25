import pytest


def test_frame_to_datauri_shrinks_and_encodes():
    np = pytest.importorskip("numpy")      # numpy/PIL live in env_isaaclab, not the uv test env
    pytest.importorskip("PIL")
    from agentbot.vla.frame_encode import frame_to_datauri

    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    uri = frame_to_datauri(img, max_side=320)
    assert uri.startswith("data:image/jpeg;base64,")
    assert len(uri) > 100


def test_frame_to_datauri_handles_batch_dim():
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from agentbot.vla.frame_encode import frame_to_datauri

    img = (np.random.rand(1, 240, 320, 3) * 255).astype(np.uint8)  # num_envs=1 leading dim
    uri = frame_to_datauri(img)
    assert uri.startswith("data:image/jpeg;base64,")
