"""Local model interface. No API or UI imports are needed by participant 2."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EvidenceFrame:
    timestamp_sec: float
    jpeg_bytes: bytes


@dataclass
class Event:
    event_type: str
    timecode_sec: float
    description: str
    evidence_frames: list[EvidenceFrame]
    rule_id: str | None = None


@dataclass
class AnalysisResult:
    verdict: Literal["violation", "no_violation"]
    events: list[Event] = field(default_factory=list)
