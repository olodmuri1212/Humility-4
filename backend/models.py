# """
# Pydantic models for the Humility Interview API
# """

# from typing import Dict, Any, Optional
# from pydantic import BaseModel

# class GenerateAudioRequest(BaseModel):
#     text: str

# class GenerateAudioResponse(BaseModel):
#     audio_b64: str

# class AnalyzeRequest(BaseModel):
#     transcript: str
#     question: Optional[str] = ""

# class AnalysisResponse(BaseModel):
#     scores: Dict[str, Dict[str, Any]]

# class TranscribeRequest(BaseModel):
#     audio_b64: str

# class TranscribeResponse(BaseModel):
#     transcript: str















# backend/models.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ========= Core agent result types =========

@dataclass
class AnalysisResult:
    """
    Canonical result produced by any agent in the system.
    """
    agent: str                    # e.g., "Humility-Scorer", "Integrity-Guard"
    score: float                  # 0..1 or 0..100 (consistent app-wide)
    label: str = ""               # optional human label ("High" / "Medium" / "Low")
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def score_pct(self) -> float:
        """Return score on [0,100] regardless of internal scale."""
        return self.score * 100.0 if self.score <= 1.0 else float(self.score)


@dataclass
class AggregateAnalysis:
    """
    Rollup across multiple agent results (a single turn or full session).
    """
    results: List[AnalysisResult] = field(default_factory=list)
    summary: str = ""
    overall_score: Optional[float] = None   # prefer normalized 0..1


# ========= Request/Response payloads =========
# Some codepaths expect either Analyze* or Analysis* naming; we provide both.

@dataclass
class AnalyzeRequest:
    candidate_id: str
    role: str
    question: str
    answer: str
    turn_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzeResponse:
    results: List[AnalysisResult] = field(default_factory=list)
    aggregate: Optional[AggregateAnalysis] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [asdict(r) for r in self.results],
            "aggregate": asdict(self.aggregate) if self.aggregate else None,
            "errors": list(self.errors),
        }


# Same idea, alternate naming used by some modules:
@dataclass
class AnalysisRequest:
    candidate_id: str
    role: str
    question: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResponse:
    results: List[AnalysisResult] = field(default_factory=list)
    aggregate: Optional[AggregateAnalysis] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [asdict(r) for r in self.results],
            "aggregate": asdict(self.aggregate) if self.aggregate else None,
            "errors": list(self.errors),
        }


# ========= Optional audio/TTS payloads (kept minimal in case other code imports them) =========

@dataclass
class GenerateAudioRequest:
    text: str
    voice_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateAudioResponse:
    sample_rate: int
    waveform: List[float]  # mono float32 list
    errors: List[str] = field(default_factory=list)


# Some projects use this alternate naming:
@dataclass
class AudioGenerateRequest:
    text: str
    voice_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioGenerateResponse:
    sample_rate: int
    waveform: List[float]
    errors: List[str] = field(default_factory=list)


__all__ = [
    # core
    "AnalysisResult",
    "AggregateAnalysis",
    # analyze/analysis pairs
    "AnalyzeRequest",
    "AnalyzeResponse",
    "AnalysisRequest",
    "AnalysisResponse",
    # audio (optional)
    "GenerateAudioRequest",
    "GenerateAudioResponse",
    "AudioGenerateRequest",
    "AudioGenerateResponse",
]
