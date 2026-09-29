"""Pipeline orchestration: sample frames, detect, validate, aggregate, report."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass

from shapely.geometry import box
from shapely.prepared import PreparedGeometry, prep

from pitch_engine.aggregation import StreamingAggregator
from pitch_engine.config import AppConfig
from pitch_engine.detectors import FieldDetector
from pitch_engine.errors import DetectorError, FeedQualityError, VideoSourceError
from pitch_engine.models import RejectReason, RunSummary
from pitch_engine.payloads import EventReport, ProgressReport
from pitch_engine.reporting import Reporter
from pitch_engine.validation import validate_polygon
from pitch_engine.video import SampledFrame, VideoFrameSource

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    summary: RunSummary
    final_report_delivered: bool


class PitchPipeline:
    """Runs one job: video in, validated boundary metrics out."""

    def __init__(self, config: AppConfig, detector: FieldDetector, reporter: Reporter) -> None:
        self._config = config
        self._detector = detector
        self._reporter = reporter

    def run(self) -> RunOutcome:
        """Run the job and tell the platform how it went.

        The failure report is sent for *any* exception (including a SIGTERM turned into
        ``KeyboardInterrupt``) and the original exception is always re-raised: reporting
        can add information but can never hide a failure.
        """
        job_id = self._config.job_id
        self._reporter.send_event(EventReport.started(job_id))
        try:
            summary = self._process()
        except (Exception, KeyboardInterrupt) as exc:
            if not self._reporter.send_event(EventReport.failed(job_id, exc)):
                log.error("failure_report_undelivered")
            raise
        delivered = self._reporter.send_event(EventReport.completed(job_id, summary))
        return RunOutcome(summary, delivered)

    def _process(self) -> RunSummary:
        cfg = self._config
        started = time.perf_counter()
        aggregator = StreamingAggregator()
        rejected: Counter[str] = Counter()
        sampled = 0

        with VideoFrameSource(cfg.video.path, cfg.sampling) as source:
            frame_box = box(0, 0, source.width, source.height)
            frame_area = frame_box.area
            inside_frame = prep(frame_box)
            log.info(
                "run_started",
                extra={
                    "video_path": cfg.video.path,
                    "video_fps": source.fps,
                    "video_frame_count": source.frame_count,
                    "frame_step": source.step,
                    "sample_strategy": source.strategy,
                },
            )
            for frame in source.frames():
                sampled += 1
                reason = self._analyse(frame, inside_frame, frame_area, aggregator)
                if reason is not None:
                    rejected[reason.value] += 1
                    log.debug(
                        "frame_rejected",
                        extra={"frame_index": frame.index, "reason": reason.value},
                    )
                if sampled % cfg.progress_every_samples == 0:
                    percent = source.percent_complete(frame.index)
                    log.info(
                        "progress",
                        extra={
                            "frames_sampled": sampled,
                            "valid_detections": aggregator.count,
                            "rejected_frames": sum(rejected.values()),
                            "percent_complete": percent,
                        },
                    )
                    self._reporter.send_progress(
                        ProgressReport(
                            job_id=cfg.job_id,
                            frames_sampled=sampled,
                            valid_detections=aggregator.count,
                            rejected_frames=sum(rejected.values()),
                            percent_complete=percent,
                        )
                    )
            summary_source = source

        if sampled == 0:
            raise VideoSourceError(f"no frames could be sampled from {cfg.video.path}")
        if summary_source.truncated:
            log.warning("stream_truncated", extra={"frames_sampled": sampled})

        rejected_total = sum(rejected.values())
        invalid_ratio = rejected_total / sampled
        elapsed = time.perf_counter() - started
        summary = RunSummary(
            job_id=cfg.job_id,
            video_path=cfg.video.path,
            video_fps=summary_source.fps,
            video_frame_count=summary_source.frame_count,
            frame_step=summary_source.step,
            sample_strategy=summary_source.strategy,
            frames_sampled=sampled,
            valid_detections=aggregator.count,
            rejected=dict(rejected),
            invalid_ratio=invalid_ratio,
            aggregate=aggregator.result(),
            stream_truncated=summary_source.truncated,
            elapsed_seconds=elapsed,
            samples_per_second=sampled / elapsed if elapsed > 0 else 0.0,
        )
        if invalid_ratio > cfg.quality.max_invalid_ratio:
            raise FeedQualityError(
                f"{rejected_total} of {sampled} sampled frames were unusable "
                f"({invalid_ratio:.0%} > allowed {cfg.quality.max_invalid_ratio:.0%}): "
                f"{dict(rejected)}"
            )
        log.info("run_completed", extra={"summary": summary.model_dump(mode="json")})
        return summary

    def _analyse(
        self,
        frame: SampledFrame,
        inside_frame: PreparedGeometry,
        frame_area: float,
        aggregator: StreamingAggregator,
    ) -> RejectReason | None:
        """Return why the frame was rejected, or ``None`` after adding it to the aggregate."""
        if frame.image is None:
            return RejectReason.UNREADABLE_FRAME
        try:
            result = self._detector.detect(frame.image)
        except Exception as exc:
            # A detector that raises is broken, not "seeing nothing": stop the run.
            raise DetectorError(f"detector failed on frame {frame.index}: {exc}") from exc
        if result.polygon is None:
            return result.reject_reason or RejectReason.NO_PITCH
        reason = validate_polygon(result.polygon, inside_frame)
        if reason is not None:
            return reason
        aggregator.add(result.polygon, frame_area)
        return None
