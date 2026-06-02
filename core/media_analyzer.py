from __future__ import annotations

import tempfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline


FACIAL_EXPRESSION_MODEL = "trpakov/vit-face-expression"


def _frame_metrics(frame: np.ndarray) -> dict[str, float]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    return {
        "brightness": round(float(np.mean(gray) / 255), 3),
        "contrast": round(float(np.std(gray) / 128), 3),
        "saturation": round(float(np.mean(hsv[:, :, 1]) / 255), 3),
        "edge_density": round(float(np.mean(edges > 0)), 3),
        "warmth": round(float((np.mean(rgb[:, :, 0]) - np.mean(rgb[:, :, 2])) / 255), 3),
    }


def _merge_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {key: round(float(np.mean([item[key] for item in items])), 3) for key in keys}


def _detect_faces(frame: np.ndarray) -> int:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return int(len(faces))


def _proxy_label(metrics: dict[str, float], face_count: int) -> str:
    if not metrics:
        return "insufficient_visual_signal"
    if metrics["edge_density"] > 0.18 and metrics["contrast"] > 0.55:
        return "visually_busy_or_high_friction"
    if metrics["brightness"] < 0.28:
        return "low_visibility_or_unclear_context"
    if metrics["saturation"] < 0.18 and metrics["contrast"] < 0.35:
        return "muted_or_low_engagement_visual_context"
    if face_count > 0:
        return "human_presence_detected"
    return "neutral_visual_context"


@lru_cache(maxsize=1)
def _expression_classifier():
    return pipeline(
        task="image-classification",
        model=FACIAL_EXPRESSION_MODEL,
        top_k=3,
    )


def _classify_expression(image: Image.Image) -> dict[str, Any]:
    predictions = _expression_classifier()(image.convert("RGB"))
    top_prediction = predictions[0] if predictions else {"label": "unknown", "score": 0.0}
    return {
        "model": FACIAL_EXPRESSION_MODEL,
        "label": str(top_prediction["label"]).lower(),
        "confidence": round(float(top_prediction["score"]), 3),
        "top_predictions": [
            {"label": str(item["label"]).lower(), "confidence": round(float(item["score"]), 3)}
            for item in predictions
        ],
    }


def _aggregate_expression_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "model": FACIAL_EXPRESSION_MODEL,
            "label": "unknown",
            "confidence": 0.0,
            "top_predictions": [],
        }

    scores: dict[str, list[float]] = {}
    for result in results:
        for item in result.get("top_predictions", []):
            scores.setdefault(item["label"], []).append(float(item["confidence"]))

    aggregated = [
        {"label": label, "confidence": round(float(np.mean(values)), 3)}
        for label, values in scores.items()
    ]
    aggregated.sort(key=lambda item: item["confidence"], reverse=True)
    top_prediction = aggregated[0]
    return {
        "model": FACIAL_EXPRESSION_MODEL,
        "label": top_prediction["label"],
        "confidence": top_prediction["confidence"],
        "top_predictions": aggregated[:3],
    }


def analyze_image_bytes(file_bytes: bytes) -> dict[str, Any]:
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    metrics = _frame_metrics(frame)
    face_count = _detect_faces(frame)
    expression = _classify_expression(image)
    return {
        "media_type": "image",
        "frame_count_sampled": 1,
        "face_count_proxy": face_count,
        "visual_metrics": metrics,
        "visual_affect_proxy": _proxy_label(metrics, face_count),
        "facial_expression": expression,
        "method": "opencv_low_level_visual_cues",
    }


def analyze_video_bytes(file_bytes: bytes, suffix: str = ".mp4", max_frames: int = 8) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    capture = cv2.VideoCapture(temp_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, frame_count // max_frames) if frame_count else 1
    metrics: list[dict[str, float]] = []
    face_counts: list[int] = []
    expression_results: list[dict[str, Any]] = []
    index = 0

    while len(metrics) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            metrics.append(_frame_metrics(frame))
            face_counts.append(_detect_faces(frame))
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            expression_results.append(_classify_expression(image))
        index += 1

    capture.release()
    Path(temp_path).unlink(missing_ok=True)

    merged_metrics = _merge_metrics(metrics)
    face_count = int(max(face_counts) if face_counts else 0)
    return {
        "media_type": "video",
        "frame_count_sampled": len(metrics),
        "face_count_proxy": face_count,
        "visual_metrics": merged_metrics,
        "visual_affect_proxy": _proxy_label(merged_metrics, face_count),
        "facial_expression": _aggregate_expression_results(expression_results),
        "method": "opencv_sampled_frame_visual_cues",
    }


def analyze_uploaded_media(file_name: str, mime_type: str, file_bytes: bytes) -> dict[str, Any] | None:
    if mime_type.startswith("image"):
        return analyze_image_bytes(file_bytes)
    if mime_type.startswith("video"):
        suffix = Path(file_name).suffix or ".mp4"
        return analyze_video_bytes(file_bytes, suffix=suffix)
    return None
