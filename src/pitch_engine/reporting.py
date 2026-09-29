"""Reporting to the platform over HTTP.

Design rule: **a reporting problem never becomes a pipeline problem.** Every method returns a
bool ("was it delivered?") and never raises, so an unreachable API cannot abort a video job. The
caller decides what an undelivered *final* report means (see the CLI exit code).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

import requests

from pitch_engine.config import ReportingConfig
from pitch_engine.payloads import EventReport, ProgressReport

log = logging.getLogger(__name__)

PROGRESS_PATH = "/api/v1/jobs/progress"
EVENTS_PATH = "/api/v1/jobs/events"


class Reporter(Protocol):
    def send_progress(self, report: ProgressReport) -> bool: ...

    def send_event(self, report: EventReport) -> bool: ...


class NullReporter:
    """Used when reporting is disabled. Nothing to deliver, so nothing can be undelivered."""

    def send_progress(self, report: ProgressReport) -> bool:
        return True

    def send_event(self, report: EventReport) -> bool:
        return True


class HttpReporter:
    def __init__(
        self,
        config: ReportingConfig,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if config.api_url is None:
            raise ValueError("HttpReporter needs reporting.api_url")
        self._config = config
        self._base_url = str(config.api_url).rstrip("/")
        self._session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._progress_failures = 0
        self._progress_disabled = False

    def send_progress(self, report: ProgressReport) -> bool:
        """Best effort: one attempt, and give up on progress after repeated failures.

        Progress is only a convenience, and each failed attempt can cost a full timeout in the
        middle of a video job, so it must not be retried aggressively.
        """
        if self._progress_disabled:
            return False
        delivered = self._post(PROGRESS_PATH, report.model_dump_json(), retries=0, kind="progress")
        if delivered:
            self._progress_failures = 0
            return True
        self._progress_failures += 1
        if self._progress_failures >= self._config.max_consecutive_progress_failures:
            self._progress_disabled = True
            log.error("progress_reporting_disabled", extra={"failures": self._progress_failures})
        return False

    def send_event(self, report: EventReport) -> bool:
        """Lifecycle events matter to the orchestrator, so they get the full retry budget."""
        return self._post(
            EVENTS_PATH,
            report.model_dump_json(),
            retries=self._config.max_retries,
            kind=report.event,
        )

    def _post(self, path: str, body: str, retries: int, kind: str) -> bool:
        url = self._base_url + path
        for attempt in range(retries + 1):
            try:
                response = self._session.post(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    timeout=self._config.timeout_seconds,
                )
            except requests.RequestException as exc:
                log.warning(
                    "report_attempt_failed",
                    extra={"kind": kind, "attempt": attempt + 1, "reason": type(exc).__name__},
                )
            else:
                if response.ok:
                    return True
                log.warning(
                    "report_attempt_rejected",
                    extra={"kind": kind, "attempt": attempt + 1, "status": response.status_code},
                )
                if response.status_code < 500:
                    return False  # 4xx: our request is wrong, retrying cannot fix it
            if attempt < retries:
                self._sleep(self._config.backoff_seconds * 2**attempt)
        log.error("report_undelivered", extra={"kind": kind, "attempts": retries + 1})
        return False


def build_reporter(config: ReportingConfig) -> Reporter:
    if not config.enabled:
        return NullReporter()
    return HttpReporter(config)