"""Pure helper: numpy RGB frame -> downsized JPEG data-uri (no IsaacLab import).

Kept separate from sim_session.py (which imports isaaclab at module top) so the Brain
side and tests can use it without a simulator. numpy/PIL are imported lazily so this
module imports even in environments that lack them (e.g. the agentbot uv env, where
numpy only lives in env_isaaclab).
"""
import base64
import io


def frame_to_datauri(img, max_side: int = 320, quality: int = 70) -> str:
    import numpy as np
    from PIL import Image

    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:      # drop a leading batch dim (num_envs=1)
        arr = arr[0]
    im = Image.fromarray(np.ascontiguousarray(arr).astype(np.uint8)).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
