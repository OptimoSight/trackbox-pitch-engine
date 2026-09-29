from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import requests
from pydantic import ValidationError

from pitch_engine.config import ReportingConfig
from pitch_engine.errors import DetectorError
from pitch_engine.models import RunSummary
from pitch_engine.payloads import EventReport, ProgressReport
from pitch_engine.reporting import HttpReporter, NullReporter, build_reporter


class FakeSession:
    """Plays back a script of status codes / exceptions."""

    def __init__(self, *script: int | Exception) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, data: str, headers: dict[str, str], timeout: float) -> Any:
        self.calls.append({"url": url, "data": data, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(ok=item < 400, status_code=item)


def _config(**overrides: Any) -> ReportingConfig:
    values: dict[str, Any] = {
        "api_url": "http://api:5000",
        "max_retries": 2,
        "backoff_seconds": 0.5,
    }
    values.update(overrides)
    return ReportingConfig(**values)


def _progress() -> ProgressReport:
    return ProgressReport(job_id="j", frames_sampled=5, valid_detections=4, rejected_frames=1)


def _reporter(session: FakeSession, sleeps: list[float] | None = None, **cfg: Any) -> HttpReporter:
    slept = sleeps if sleeps is not None else []
    return HttpReporter(_config(**cfg), session=session, sleep=slept.append)


def test_event_is_posted_to_the_events_endpoint_as_json() -> None:
    session = FakeSession(200)
    assert _reporter(session).send_event(EventReport.started("job-1")) is True
    call = session.calls[0]
    assert call["url"] == "http://api:5000/api/v1/jobs/events"
    assert '"event":"started"' in call["data"]
    assert '"job_id":"job-1"' in call["data"]


def test_progress_goes_to_the_progress_endpoint() -> None:
    session = FakeSession(200)
    assert _reporter(session).send_progress(_progress()) is True
    assert session.calls[0]["url"].endswith("/api/v1/jobs/progress")


def test_server_errors_are_retried_with_backoff_then_succeed() -> None:
    session = FakeSession(503, 500, 200)
    sleeps: list[float] = []
    assert _reporter(session, sleeps).send_event(EventReport.started("j")) is True
    assert len(session.calls) == 3
    assert sleeps == [0.5, 1.0]


def test_connection_errors_are_retried_and_never_raise() -> None:
    boom = requests.ConnectionError("down")
    session = FakeSession(boom, boom, boom)
    assert _reporter(session).send_event(EventReport.started("j")) is False
    assert len(session.calls) == 3


def test_client_errors_are_not_retried() -> None:
    session = FakeSession(422)
    assert _reporter(session).send_event(EventReport.started("j")) is False
    assert len(session.calls) == 1


def test_progress_is_single_attempt_and_switches_off_after_repeated_failures() -> None:
    boom = requests.Timeout("slow")
    session = FakeSession(boom, boom, boom)
    reporter = _reporter(session, max_consecutive_progress_failures=3)
    assert [reporter.send_progress(_progress()) for _ in range(5)] == [False] * 5
    assert len(session.calls) == 3  # attempts 4 and 5 were skipped, not sent


def test_final_events_still_go_out_after_progress_is_switched_off() -> None:
    boom = requests.Timeout("slow")
    session = FakeSession(boom, 200)
    reporter = _reporter(session, max_consecutive_progress_failures=1)
    reporter.send_progress(_progress())
    assert reporter.send_event(EventReport.started("j")) is True


def test_success_resets_the_progress_failure_counter() -> None:
    boom = requests.Timeout("slow")
    session = FakeSession(boom, 200, boom, 200)
    reporter = _reporter(session, max_consecutive_progress_failures=2)
    results = [reporter.send_progress(_progress()) for _ in range(4)]
    assert results == [False, True, False, True]


def test_null_reporter_and_factory() -> None:
    assert NullReporter().send_event(EventReport.started("j")) is True
    assert isinstance(build_reporter(ReportingConfig(enabled=False)), NullReporter)
    assert isinstance(build_reporter(_config()), HttpReporter)


def test_api_url_is_required_when_enabled() -> None:
    with pytest.raises(ValidationError, match="api_url"):
        ReportingConfig(enabled=True)
    with pytest.raises(ValidationError):
        ReportingConfig(api_url="not a url")


def test_payloads_are_validated_before_sending() -> None:
    with pytest.raises(ValidationError):
        ProgressReport(job_id="j", frames_sampled=-1, valid_detections=0, rejected_frames=0)
    with pytest.raises(ValidationError):
        ProgressReport(
            job_id="j",
            frames_sampled=1,
            valid_detections=0,
            rejected_frames=0,
            percent_complete=101,
        )
    with pytest.raises(ValidationError, match="summary"):
        EventReport(job_id="j", event="completed")
    with pytest.raises(ValidationError, match="error"):
        EventReport(job_id="j", event="failed")


def test_failed_event_carries_the_exit_code_of_the_error() -> None:
    report = EventReport.failed("j", DetectorError("bad frame"))
    assert report.error is not None
    assert (report.error.type, report.error.message, report.error.exit_code) == (
        "DetectorError",
        "bad frame",
        1,
    )
    assert EventReport.failed("j", KeyboardInterrupt()).error.exit_code == 130  # type: ignore[union-attr]


def test_completed_event_round_trips_a_summary() -> None:
    summary = RunSummary(
        job_id="j",
        video_path="v.mp4",
        video_fps=30.0,
        video_frame_count=300,
        frame_step=30,
        sample_strategy="seek",
        frames_sampled=10,
        valid_detections=0,
        rejected={"no_pitch": 10},
        invalid_ratio=1.0,
        aggregate=None,
        stream_truncated=False,
        elapsed_seconds=1.0,
        samples_per_second=10.0,
    )
    text = EventReport.completed("j", summary).model_dump_json()
    assert EventReport.model_validate_json(text).summary == summary
