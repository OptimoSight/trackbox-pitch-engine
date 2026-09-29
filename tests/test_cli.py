from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from pitch_engine.cli import main

WriteConfig = Callable[..., Path]
DEAD_URL = "http://127.0.0.1:9"  # nothing listens on the discard port


class _Api:
    def __init__(self) -> None:
        self.received: list[tuple[str, dict[str, Any]]] = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                api.received.append((self.path, json.loads(body)))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args: Any) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def events(self) -> list[str]:
        return [b["event"] for p, b in self.received if p.endswith("/events")]

    def progress(self) -> list[dict[str, Any]]:
        return [b for p, b in self.received if p.endswith("/progress")]


@pytest.fixture
def api() -> Iterator[_Api]:
    server = _Api()
    yield server
    server.server.shutdown()


def test_bad_config_exits_2_before_doing_any_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"video": {"path": "x.mp4"}, "detector": {"type": "nope"}}))
    assert main(["--config", str(bad)]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_missing_video_exits_3_and_reports_failure(
    write_config: WriteConfig, api: _Api, tmp_path: Path
) -> None:
    config = write_config(
        video={"path": str(tmp_path / "gone.mp4")},
        reporting={"enabled": True, "api_url": api.url},
    )
    assert main(["--config", str(config)]) == 3
    assert api.events() == ["started", "failed"]
    failed = next(b for p, b in api.received if b.get("event") == "failed")
    assert failed["error"]["type"] == "VideoSourceError"
    assert failed["error"]["exit_code"] == 3


def test_successful_run_reports_started_progress_completed(
    write_config: WriteConfig, api: _Api
) -> None:
    config = write_config(
        reporting={"enabled": True, "api_url": api.url},
        progress_every_samples=5,
    )
    assert main(["--config", str(config)]) == 0
    assert api.events() == ["started", "completed"]
    assert [p["frames_sampled"] for p in api.progress()] == [5, 10]
    completed = next(b for p, b in api.received if b.get("event") == "completed")
    assert completed["summary"]["frames_sampled"] == 10
    assert completed["summary"]["aggregate"]["area_ratio_mean"] > 0.4


def test_unreachable_api_does_not_fail_the_pipeline_but_is_not_silent(
    write_config: WriteConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(
        reporting={
            "enabled": True,
            "api_url": DEAD_URL,
            "max_retries": 1,
            "backoff_seconds": 0.0,
            "timeout_seconds": 1.0,
        },
    )
    code = main(["--config", str(config)])
    out = capsys.readouterr().out
    assert code == 5  # the video job succeeded, but the outcome could not be delivered
    assert '"event": "run_completed"' in out  # the pipeline itself finished normally
    assert '"event": "report_undelivered"' in out


def test_reporting_can_be_disabled(write_config: WriteConfig) -> None:
    assert main(["--config", str(write_config())]) == 0


def test_feed_quality_failure_exits_4(write_config: WriteConfig, api: _Api) -> None:
    config = write_config(
        quality={"max_invalid_ratio": 0.0},
        detector={"min_area_px": 10_000_000},  # nothing can be big enough: every frame rejected
        reporting={"enabled": True, "api_url": api.url},
    )
    assert main(["--config", str(config)]) == 4
    assert api.events() == ["started", "failed"]


def test_generate_video_flag_creates_a_missing_feed(
    write_config: WriteConfig, tmp_path: Path
) -> None:
    target = tmp_path / "generated.mp4"
    config = write_config(video={"path": str(target)}, sampling={"sample_fps": 0.25})
    assert main(["--config", str(config), "--generate-video"]) == 0
    assert target.exists()
