"""Error taxonomy.

Every failure the pipeline raises on purpose is a ``PitchEngineError`` carrying the process exit
code an orchestrator can act on. Anything that is *not* a ``PitchEngineError`` is a bug and is
reported as such (exit code 1) instead of being absorbed.
"""

from __future__ import annotations


class PitchEngineError(Exception):
    """Base class for expected, fatal failures."""

    exit_code = 1


class ConfigError(PitchEngineError):
    """Configuration is missing, unparsable or invalid. Raised at load time, before any work."""

    exit_code = 2


class VideoSourceError(PitchEngineError):
    """The video cannot be opened or its metadata is unusable (fatal, nothing to process)."""

    exit_code = 3