from __future__ import annotations

from .schemas import EmotionSignal, Evidence, Interview, TranscriptSegment, clamp


EMOTION_KEYWORDS = {
    "frustrated": ["frustrating", "frustrated", "annoying", "stuck", "waste"],
    "confused": ["confused", "not sure", "unclear", "lost", "what this means"],
    "hesitant": ["maybe", "i guess", "not sure", "probably", "hesitant"],
    "relieved": ["relieved", "finally", "that helps", "clear now", "makes sense"],
    "overwhelmed": ["too much", "overwhelmed", "a lot", "too many", "hard to follow"],
}

CANONICAL_VALENCE = {
    "frustrated": "negative",
    "confused": "negative",
    "hesitant": "mixed",
    "relieved": "positive",
    "overwhelmed": "negative",
}

MODEL_EMOTION_MAP = {
    "angry": "frustrated",
    "disgust": "frustrated",
    "fear": "overwhelmed",
    "happy": "relieved",
    "neutral": "hesitant",
    "sad": "frustrated",
    "surprise": "confused",
}


def _cue_value(segment: TranscriptSegment, cue_path: str, default: float = 0.0) -> float:
    data = segment.cues
    for part in cue_path.split("."):
        if not isinstance(data, dict) or part not in data:
            return default
        data = data[part]
    try:
        return float(data)
    except (TypeError, ValueError):
        return default


def _keyword_label(text: str) -> str | None:
    lower = text.lower()
    for label, keywords in EMOTION_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return label
    return None


def _model_expression_label(segment: TranscriptSegment) -> tuple[str | None, float]:
    expression = segment.cues.get("media_model", {}).get("facial_expression", {})
    label = str(expression.get("label", "")).lower()
    confidence = float(expression.get("confidence", 0.0) or 0.0)
    return MODEL_EMOTION_MAP.get(label), confidence


def infer_segment_emotion(segment: TranscriptSegment) -> EmotionSignal:
    text_label = _keyword_label(segment.text)
    model_label, model_confidence = _model_expression_label(segment)
    text_sentiment = _cue_value(segment, "text.sentiment", 0.0)
    pause_ratio = _cue_value(segment, "audio.pause_ratio", 0.0)
    pitch_delta = abs(_cue_value(segment, "audio.pitch_delta", 0.0))
    facial_tension = _cue_value(segment, "facial.brow_tension", 0.0)
    smile_score = _cue_value(segment, "facial.smile_score", 0.0)
    uncertainty_words = _cue_value(segment, "text.uncertainty_words", 0.0)
    relief_words = _cue_value(segment, "text.relief_words", 0.0)

    if text_label:
        label = text_label
    elif model_label and model_confidence >= 0.45:
        label = model_label
    elif relief_words > 0 or text_sentiment > 0.45 or smile_score > 0.62:
        label = "relieved"
    elif facial_tension + pause_ratio + uncertainty_words > 1.45:
        label = "overwhelmed"
    elif uncertainty_words > 0 or pause_ratio > 0.25:
        label = "hesitant"
    elif text_sentiment < -0.35 or facial_tension > 0.7:
        label = "frustrated"
    else:
        label = "confused"

    intensity = clamp(
        0.28
        + 0.24 * facial_tension
        + 0.18 * pause_ratio
        + 0.16 * pitch_delta
        + 0.12 * uncertainty_words
        + 0.10 * abs(text_sentiment)
    )
    if label == "relieved":
        intensity = clamp(0.35 + 0.35 * smile_score + 0.2 * relief_words + 0.1 * text_sentiment)

    evidence = [
        Evidence(
            segment_id=segment.segment_id,
            quote=segment.text,
            source="transcript",
            timestamp_start=segment.timestamp_start,
            timestamp_end=segment.timestamp_end,
            cue="verbatim transcript",
            confidence=0.95,
        )
    ]
    if pause_ratio > 0.2:
        evidence.append(
            Evidence(
                segment_id=segment.segment_id,
                quote=f"pause_ratio={pause_ratio:.2f}",
                source="mock_audio_metadata",
                timestamp_start=segment.timestamp_start,
                timestamp_end=segment.timestamp_end,
                cue="long pauses can indicate hesitation in this task context",
                confidence=0.68,
            )
        )
    if facial_tension > 0.55:
        evidence.append(
            Evidence(
                segment_id=segment.segment_id,
                quote=f"brow_tension={facial_tension:.2f}",
                source="mock_facial_metadata",
                timestamp_start=segment.timestamp_start,
                timestamp_end=segment.timestamp_end,
                cue="brow tension is treated as a weak friction signal, not a diagnosis",
                confidence=0.62,
            )
        )
    if model_label and model_confidence > 0:
        evidence.append(
            Evidence(
                segment_id=segment.segment_id,
                quote=f"facial_expression_model={model_label}, confidence={model_confidence:.2f}",
                source="facial_expression_model",
                timestamp_start=segment.timestamp_start,
                timestamp_end=segment.timestamp_end,
                cue="visible facial expression classification from an uploaded image/video; not a psychological diagnosis",
                confidence=min(0.85, model_confidence),
            )
        )

    rationale = (
        f"Segment uses language/cues consistent with '{label}' in this interaction context. "
        "This is an open-vocabulary expression signal, not a psychological diagnosis."
    )
    return EmotionSignal(
        segment_id=segment.segment_id,
        timestamp_start=segment.timestamp_start,
        timestamp_end=segment.timestamp_end,
        label=label,
        canonical_label=label,
        valence=CANONICAL_VALENCE.get(label, "mixed"),
        intensity=round(intensity, 3),
        confidence=round(clamp(0.55 + intensity * 0.35), 3),
        evidence=evidence,
        rationale=rationale,
    )


def build_emotion_timeline(interview: Interview) -> list[EmotionSignal]:
    return [infer_segment_emotion(segment) for segment in interview.transcript_segments]
