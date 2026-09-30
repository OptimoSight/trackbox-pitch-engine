# Decisions

This document explains what was wrong with the v0.1 prototype, what I changed, and the trade-offs
behind each choice. Numbers come from running the code on the synthetic feed
(1800 frames, 1280x720, 30 fps).

## What was wrong with the prototype

| # | Problem in `synthetic_field_prototype.py` | Consequence | Fixed by |
|---|---|---|---|
| 1 | Everything in one script | Cannot be reused or unit-tested | `pitch_engine` package + thin `main.py` |
| 2 | Raw `CONFIG` dict; `.get(..., "soccer")` defaults; half the keys never read | Typos and bad values pass silently; the config says `football`, the code silently used `soccer` | Pydantic models, `extra="forbid"`, fail at load time |
| 3 | Detection hard-wired into the analyzer | Every new sport/model means editing the pipeline | `FieldDetector` protocol (the one seam) |
| 4 | `RETR_EXTERNAL` on a green mask | **The "boundary" was the whole frame on 1776 of 1800 frames** (`(0,0,1279,719)`, area 919601), so the reported metric was a constant, and camera cuts, close-ups and noise were indistinguishable from real detections | Detector picks the largest green region that is not the whole frame |
| 5 | `time.sleep(0.005)` per frame, every frame decoded and converted | Runtime grows linearly with file length | Time-based sampling with seeking |
| 6 | Frame rectangle rebuilt per frame; `np.array` bounds rebuilt per frame; 1280x720 hard-coded | Repeated work, wrong for other resolutions | Computed once from the real frame size / once in `__init__` |
| 7 | `except Exception: pass` around contour extraction | Any bug looked like "no pitch"; a real failure could never surface | Removed. Expected outcomes are data, unexpected ones raise |
| 8 | All polygons kept in a list | Memory grows with video length | Streaming aggregator, constant memory |
| 9 | `print()` and `return` on failure (exit code 0) | Nobody watching a batch job could tell it failed | JSON logs, typed errors, distinct exit codes |
| 10 | Failed frames silently dropped, no aggregate output | No idea how much of the feed was usable | Per-reason rejection counts, validated aggregate |
| 11 | Comment says "generate if it doesn't exist", code always regenerates | Misleading, wasteful | `--generate-video` only generates when missing |
| 12 | No reporting to the platform | Orchestrator cannot see progress or outcome | HTTP reporter with validated payloads |

## 1. Assumptions and open questions

Assumptions I made:

- **The feed is a normal H.264/MPEG-4 file with a valid frame rate and frame count.** If the
  frame rate is missing or the size is zero the run fails immediately. If only the frame count is
  missing, the pipeline falls back to sequential decoding and logs a warning.
- **A boundary changes slowly**, so looking at about one frame per second is enough to
  characterise a match. This is configurable (`sampling.sample_fps`). A real broadcast with fast
  cuts may need more.
- **Camera cuts, close-ups and noise are normal, not errors.** They are counted, not fatal, up to
  `quality.max_invalid_ratio` (default 50 %). Beyond that the feed is considered unusable.
- **"Valid" means**: the detector found a region, the polygon is a valid, non-empty shape, and it
  lies inside the frame.
- **The mock API accepts any JSON**, so I defined the payload shape myself
  (`payloads.py`): `ProgressReport` for `/progress`, `EventReport` (`started` / `completed` /
  `failed`) for `/events`.

What I would ask the product/ML team before production:

1. What does the crop step need from the boundary: the per-frame polygon, the envelope, or a
   smoothed track over time? I output area statistics, mean bounds and envelope bounds.
2. What is an acceptable invalid-frame ratio, and should a bad *segment* (say 30 s of close-ups)
   fail the job even if the overall ratio is fine?
3. Is the feed ever live or truncated? Then "frame count unknown" is the normal case, not the edge.
4. Should a run that produced results but could not report them be retried by the orchestrator
   (exit code 5) or treated as success?
5. Which real detector replaces the colour threshold, and what are its latency and memory
   budgets? That decides whether frames should be batched or the work parallelised.
6. The synthetic feed shows a white line on green. Real pitches are not like that, so the
   placeholder detector's thresholds are illustrative only.

## 2. Validation strictness vs. fallback

**Fail fast (nothing is guessed):**

- Configuration: unknown keys, wrong types, out-of-range values, unparsable file, invalid URL, a
  reporting section that is enabled without a URL. `detector.type` and `video.path` have no
  default on purpose. All problems are reported together, before any video is opened (exit 2).
- Video: file missing, cannot be opened, unusable fps or size, a start time past the end of the
  video (exit 3).
- A detector that raises is a bug, not "no pitch", and stops the run (`DetectorError`).
- Decode failures in a row (`sampling.max_consecutive_read_failures`) stop the run
  (`StreamReadError`).
- Too many unusable frames stops the run (`FeedQualityError`, exit 4).
- The old prototype keys (`confidence_threshold`, `debug_mode`, `crop_search`, ...) are rejected
  rather than ignored. They were never read, so keeping them would only mislead.

**Sensible fallbacks (each one logged):**

- Unknown frame count: seek needs an end frame, so it falls back to sequential decoding.
- A single frame that cannot be decoded, or has no pitch: counted by reason and skipped.
- Progress reports failing: after N consecutive failures progress reporting switches itself off.
  Progress is a convenience and each attempt can cost a full timeout in the middle of a job.

**Two independent failure domains (Part 4):**

| Video pipeline | Report delivered | Result |
|---|---|---|
| OK | yes | exit 0 |
| OK | **no** | exit **5**, the job is not marked failed, the summary is in the logs |
| fails | yes | `failed` event with error type and exit code, exit code of the error |
| fails | no | exit code of the error, plus an error log |

The reporter never raises, so an unreachable API can never abort a video job. The pipeline
always re-raises the original exception after sending the failure report, so a failure can never
be hidden by the reporting layer. Events are retried with exponential backoff on connection
errors and 5xx. A 4xx is not retried, because retrying cannot fix a wrong request.

**Rejected frames are never mixed into metrics.** The aggregate only ever receives valid
polygons, and with zero detections it is `null`, not a row of zeros.

## 3. Performance trade-offs

- **Sampling is the main saving.** Default is 1 frame per second (every 30th frame).
  Prototype: ~18.9 s for 1800 frames. New pipeline: ~1 s for 60 sampled frames (both measured in
  the same sandbox; absolute numbers depend on the machine, the ratio is what matters).
- **`seek` vs `sequential`.** I measured both (1800 / 7200-frame files):

  | Every Nth frame | sequential (`grab`) | seek |
  |---|---|---|
  | 30 (1 fps) | 0.90 s / 3.6 s | 0.73 s / 2.8 s |
  | 150 | 0.89 s / 3.7 s | 0.15 s / 0.61 s |
  | 600 | 0.83 s / 3.8 s | 0.03 s / 0.14 s |

  `sequential` must decode the whole file, so it is linear in the file length. `seek` is linear in
  the number of frames *sampled*: quadrupling the file at the same stride quadruples the samples,
  but sampling the same number of frames costs the same on a long file as on a short one. For
  very dense sampling (stride shorter than the codec's key-frame interval) seeking loses its
  advantage, so `sequential` stays available. Seeking accuracy depends on the codec and
  container, which is why it is a setting rather than hard-wired.
- **`start_seconds` / `duration_seconds`** limit the run to the part of the video that matters.
- **Repeated calculations** are done once: the frame rectangle (and its prepared geometry),
  the HSV bounds arrays, and the frame area.
- **Detection scale** (`detector.scale`, default 1.0) trades accuracy for speed by detecting on
  a downscaled frame. On the synthetic pitch, scale 0.5 keeps area within 3 % of full size
  (covered by a test). I left the default at full size because accuracy is the safer default
  until the real detector's cost is known.
- **Synchronous processing.** I did not add threads or processes. The saving from sampling is
  larger and the code stays simple. With a real model the next step would be batching frames
  or running detection in a worker pool.
- **Streaming aggregation** keeps memory constant however long the feed is, at the cost of not
  having every polygon available afterwards. If the crop step needs them, they should be
  written to storage rather than kept in memory.
- **Progress is sent synchronously** with a single attempt. A background thread would avoid
  stalling the loop on a slow API but adds concurrency I did not think was justified here.

## 4. AI/LLM disclosure

I used Claude (Anthropic) heavily for this exercise, and I want to be precise about who did what.

**Claude did most of the engineering work.** From the assignment and the starter code it:

- explained the prototype to me and listed what was wrong with it;
- designed the package structure, the configuration model, the detector seam, the frame sampling,
  the error handling and the reporting layer;
- wrote the implementation, the 70 tests, the Dockerfile, the compose changes, the CI workflow,
  this document and the README;
- broke the result into 20 commits on 9 branches so the history shows the work step by step;
- ran the tests, `ruff` and `mypy` in its own sandbox, ran the pipeline against the provided
  mock API, and took the measurements quoted in this document (the sampling benchmark, the
  prototype's full-frame result, the runtime comparison).

**My part.** I supplied the assignment and starter code, asked for everything to be explained in
Bangla (my first language) so I understood it before using it, and made the decision to build
the solution this way. I applied each step to my repository, ran the checks, made the commits,
opened and merged the pull requests, and pushed. I am responsible for what is submitted here and
I can explain it. I did not write the code line by line myself. I run docker and test it.

**What the assistant could not verify, and what I checked myself.** Claude's environment had no
Docker, so it could only review the Dockerfile, the compose changes and the CI Docker job by
reading them. I ran `docker compose up --build --abort-on-container-exit` on my machine. The
first run failed because `requirements.txt` still listed the full `opencv-python`, which needs
GUI system libraries the slim image does not have (`libxcb.so.1` was missing). I switched it to
`opencv-python-headless`, and the second run finished with exit code 0 and delivered the
`started`, progress and `completed` reports to the mock API. The CI Docker job is verified only
once it has run on GitHub.


The commit history therefore reflects the order in which I applied and reviewed the work, not
the order in which the code was originally written.

## What I did not do (and would do next)

- A real segmentation model behind `FieldDetector` and the crop-layout step the README mentions.
- Temporal smoothing and detection of bad *segments* rather than only an overall ratio.
- Async or background progress reporting; a persistent outbox so an undelivered final report can
  be retried after the job has exited.
- An end-to-end test in CI that runs `docker compose up` against the mock API.
- Metrics/tracing (Prometheus, OpenTelemetry) on top of the structured logs.
- Handling of a container stop signal beyond turning `SIGTERM` into a `failed` report.