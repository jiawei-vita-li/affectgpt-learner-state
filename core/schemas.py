from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Valence = Literal["positive", "negative", "mixed", "neutral"]
Severity = Literal["low", "medium", "high"]


@dataclass
class Evidence:
    segment_id: str
    quote: str
    source: str
    timestamp_start: str
    timestamp_end: str
    cue: str
    confidence: float


@dataclass
class TranscriptSegment:
    segment_id: str
    timestamp_start: str
    timestamp_end: str
    speaker: str
    text: str
    context: dict[str, Any]
    cues: dict[str, Any]


@dataclass
class Interview:
    interview_id: str
    project_id: str
    title: str
    participant: dict[str, Any]
    session: dict[str, Any]
    research_goals: list[str]
    transcript_segments: list[TranscriptSegment]


@dataclass
class EmotionSignal:
    segment_id: str
    timestamp_start: str
    timestamp_end: str
    label: str
    canonical_label: str
    valence: Valence
    intensity: float
    confidence: float
    evidence: list[Evidence]
    rationale: str


@dataclass
class PainPoint:
    pain_point_id: str
    title: str
    description: str
    affected_user_segments: list[str]
    journey_stage: str
    product_area: str
    severity: Severity
    frequency_note: str
    evidence: list[Evidence]
    emotion_signals: list[str]
    recommended_next_step: str


@dataclass
class ProductOpportunity:
    opportunity_id: str
    title: str
    description: str
    suggested_action: str
    success_metric: str
    supporting_pain_points: list[str]
    confidence: float


@dataclass
class StructuredRationale:
    recognition: str
    evidence_attribution: list[str]
    product_suggestion: str
    safety_note: str


@dataclass
class InsightReport:
    interview_id: str
    top_pain_points: list[PainPoint] = field(default_factory=list)
    opportunities: list[ProductOpportunity] = field(default_factory=list)
    rationales: list[StructuredRationale] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def parse_interview(raw: dict[str, Any]) -> Interview:
    segments = [
        TranscriptSegment(
            segment_id=item["segment_id"],
            timestamp_start=item["timestamp_start"],
            timestamp_end=item["timestamp_end"],
            speaker=item.get("speaker", "participant"),
            text=item["text"],
            context=item.get("context", {}),
            cues=item.get("cues", {}),
        )
        for item in raw["transcript_segments"]
    ]
    return Interview(
        interview_id=raw["interview_id"],
        project_id=raw["project_id"],
        title=raw["title"],
        participant=raw["participant"],
        session=raw["session"],
        research_goals=raw.get("research_goals", []),
        transcript_segments=segments,
    )

