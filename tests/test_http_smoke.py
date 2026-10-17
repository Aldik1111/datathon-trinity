"""Live HTTP check: a fresh server and a spawned worker, synthetic MOV only."""
import base64
import multiprocessing
import socket
import time

import cv2
import numpy as np
import requests

from src.api.settings import Settings


def serve(settings, port):
    import uvicorn
    from src.api.app import create_app
    uvicorn.run(create_app(settings), host="127.0.0.1", port=port, access_log=False, log_level="critical")


def test_live_http_to_spawned_worker(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings = Settings(api_key="synthetic-http-key", upload_dir=tmp_path / "uploads", max_upload_mb=1, analyzer_mode="demo")
    process = multiprocessing.get_context("spawn").Process(target=serve, args=(settings, port))
    process.start()
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                if requests.get(base + "/healthz", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            assert time.monotonic() < deadline, "Synthetic HTTP server did not start"
            time.sleep(.1)
        path = tmp_path / "synthetic.mov"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
        assert writer.isOpened()
        for value in (0, 50, 100):
            writer.write(np.full((48, 64, 3), value, np.uint8))
        writer.release()
        headers = {"X-API-Key": settings.api_key}
        with path.open("rb") as source:
            response = requests.post(base + "/api/v1/analyze", headers=headers, files={"file": ("synthetic.mov", source, "video/quicktime")}, timeout=5)
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 15
        while True:
            result = requests.get(base + f"/api/v1/jobs/{job_id}", headers=headers, timeout=3).json()
            if result["status"] != "processing":
                break
            assert time.monotonic() < deadline
            time.sleep(.1)
        assert result["status"] == "done"
        assert result["demo"] is True
        image = result["violations"][0]["evidence_frames"][0]
        assert cv2.imdecode(np.frombuffer(base64.b64decode(image["image_base64"]), np.uint8), cv2.IMREAD_COLOR) is not None
        assert list(settings.upload_dir.glob("*.mov")) == []
        assert list(settings.upload_dir.glob("*.worker.json")) == []
    finally:
        process.terminate()
        process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join(timeout=3)
        process.close()
