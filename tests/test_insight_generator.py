import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.emotion_timeline import build_emotion_timeline
from core.insight_generator import generate_report
from core.schemas import parse_interview


def load_first_interview():
    data_path = PROJECT_ROOT / "data" / "sample_interviews.json"
    with data_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    return parse_interview(raw[0])


class InsightGeneratorTest(unittest.TestCase):
    def test_parse_interview_segments(self):
        interview = load_first_interview()

        self.assertEqual(interview.interview_id, "int_onboarding_001")
        self.assertEqual(len(interview.transcript_segments), 4)
        self.assertEqual(interview.transcript_segments[0].segment_id, "seg_001")

    def test_open_vocabulary_emotion_timeline(self):
        interview = load_first_interview()
        timeline = build_emotion_timeline(interview)
        labels = {signal.label for signal in timeline}

        self.assertIn("overwhelmed", labels)
        self.assertIn("relieved", labels)
        self.assertTrue(all(0 <= signal.intensity <= 1 for signal in timeline))
        self.assertTrue(all(signal.evidence for signal in timeline))

    def test_report_contains_evidence_grounded_pain_points(self):
        interview = load_first_interview()
        report = generate_report(interview)

        self.assertTrue(report.top_pain_points)
        self.assertTrue(report.opportunities)
        for pain_point in report.top_pain_points:
            self.assertTrue(pain_point.evidence)
            self.assertTrue(pain_point.recommended_next_step)

    def test_structured_rationale_has_safety_boundary(self):
        interview = load_first_interview()
        report = generate_report(interview)

        self.assertTrue(report.rationales)
        first = report.rationales[0]
        self.assertIn("Detected", first.recognition)
        self.assertTrue(first.evidence_attribution)
        self.assertIn("not a psychological diagnosis", first.safety_note)

    def test_metrics_include_required_product_and_quality_metrics(self):
        interview = load_first_interview()
        report = generate_report(interview)

        expected = {
            "activation",
            "time_to_insight",
            "evidence_coverage",
            "insight_usefulness",
            "evidence_grounding",
            "emotion_consistency",
            "actionability",
        }
        self.assertTrue(expected.issubset(report.metrics.keys()))


if __name__ == "__main__":
    unittest.main()
