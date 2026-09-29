# Pitch Boundary Engine

Production rewrite of the v0.1 pitch-boundary prototype. It reads match video, finds the playing
field boundary on sampled frames, aggregates the valid detections into metrics for the
downstream crop-layout step, and reports progress and outcome to the platform API.

See [DECISIONS.md](DECISIONS.md) for what was wrong with the prototype and why each design choice
was made.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python main.py --generate-video                # generate the synthetic feed if missing, then run
python main.py --config config/default.json    # run on an existing video
pytest                                         # tests
ruff check . && ruff format --check . && mypy src
```

With the reporting service (Docker):

```bash
docker compose up --build --abort-on-container-exit
curl localhost:5000/api/v1/jobs/events         # what the runner reported
```

Logs are JSON, one object per line, on stdout. The final `run_completed` event carries the whole
summary.

## Layout

```
main.py                    thin entry point
config/default.json        default configuration
src/pitch_engine/
  config.py                validated configuration (fail fast)
  errors.py                error taxonomy + exit codes
  models.py                DetectionResult, RunSummary, AggregateMetrics
  detectors/               FieldDetector protocol + implementations (the one seam)
  video.py                 time-based frame sampling
  validation.py            detector-independent polygon checks
  aggregation.py           streaming statistics over valid detections
  pipeline.py              orchestration
  payloads.py              validated wire models
  reporting.py             HTTP reporter (never raises)
  logging_setup.py         structured logging
  cli.py                   argument parsing and exit codes
tests/
```

## Configuration

`config/default.json`, validated at startup. Unknown keys are errors. Environment overrides
(also validated): `VIDEO_PATH`, `JOB_ID`, `LOG_LEVEL`, `MOCK_API_URL`.

| Section | Key | Meaning |
|---|---|---|
| `video` | `path` | input video (required) |
| `detector` | `type` | implementation to use (required, `color_threshold`) |
| `detector` | `lower_hsv`, `upper_hsv`, `min_area_px`, `max_area_ratio`, `scale` | detector tuning |
| `sampling` | `sample_fps` | frames analysed per second of video |
| `sampling` | `strategy` | `seek` (cost follows samples) or `sequential` (decodes everything) |
| `sampling` | `start_seconds`, `duration_seconds` | limit the analysed window |
| `sampling` | `max_consecutive_read_failures` | stop when decoding keeps failing |
| `quality` | `max_invalid_ratio` | fail if more than this share of sampled frames is unusable |
| `reporting` | `enabled`, `api_url`, `timeout_seconds`, `max_retries`, `backoff_seconds` | platform API |
| `logging` | `level`, `format` | `json` or `text` |
| top level | `job_id`, `progress_every_samples` | run identity and progress cadence |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success, final report delivered |
| 1 | unexpected error (a bug) |
| 2 | invalid configuration |
| 3 | video cannot be opened / decoded |
| 4 | too many unusable frames |
| 5 | job succeeded but the final report could not be delivered |
| 130 | interrupted (SIGTERM / Ctrl-C) |

## Adding a detector

Implement `detect(frame) -> DetectionResult` (see `detectors/base.py`), add its config model, and
map it in `detectors/build_detector`. The pipeline does not change.

## Reporting contract

`POST /api/v1/jobs/progress` with `ProgressReport`; `POST /api/v1/jobs/events` with
`EventReport` (`started`, `completed` with a `summary`, `failed` with an `error`). Both are
pydantic models in `payloads.py`.