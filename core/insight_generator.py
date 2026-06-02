from __future__ import annotations

from collections import Counter
from statistics import mean

from .emotion_timeline import build_emotion_timeline
from .schemas import (
    Evidence,
    InsightReport,
    Interview,
    PainPoint,
    ProductOpportunity,
    StructuredRationale,
)


PAIN_POINT_RULES = {
    "onboarding": {
        "title": "Users lack confidence during onboarding setup",
        "title_zh": "用户在完成引导设置后仍缺乏确认感",
        "description": "Users can complete setup steps but still feel uncertain about whether the product is configured correctly.",
        "description_zh": "用户虽然能完成设置流程，但仍不确定产品是否已经被正确配置。",
        "recommended_next_step": "Add a confirmation summary with completed steps, remaining optional actions, and a clear next action.",
        "recommended_next_step_zh": "增加确认摘要，展示已完成步骤、剩余可选动作和明确的下一步。",
    },
    "dashboard": {
        "title": "Users struggle to interpret dashboard status",
        "title_zh": "用户难以理解仪表盘状态",
        "description": "Users see metrics or states but cannot tell which action matters next.",
        "description_zh": "用户能看到指标或状态，但不知道下一步最重要的动作是什么。",
        "recommended_next_step": "Add plain-language status explanations and prioritized next-step prompts.",
        "recommended_next_step_zh": "增加通俗状态解释，并给出按优先级排序的下一步提示。",
    },
    "collaboration": {
        "title": "Users hesitate before sharing work with teammates",
        "title_zh": "用户在分享给团队前存在犹豫",
        "description": "Users worry that generated outputs may be incomplete or hard to explain to collaborators.",
        "description_zh": "用户担心生成结果不完整，或难以向协作者解释其依据。",
        "recommended_next_step": "Provide share-ready evidence snippets and a reviewer-facing summary.",
        "recommended_next_step_zh": "提供可直接分享的证据片段，以及面向 reviewer 的摘要说明。",
    },
}


def _segment_by_id(interview: Interview) -> dict[str, object]:
    return {segment.segment_id: segment for segment in interview.transcript_segments}


def _pain_area(signal_label: str, product_area: str) -> str:
    if "dashboard" in product_area:
        return "dashboard"
    if "team" in product_area or "share" in product_area:
        return "collaboration"
    return "onboarding"


def generate_pain_points(interview: Interview) -> list[PainPoint]:
    timeline = build_emotion_timeline(interview)
    segments = _segment_by_id(interview)
    grouped: dict[str, list] = {}
    for signal in timeline:
        segment = segments[signal.segment_id]
        product_area = segment.context.get("product_area", "onboarding")
        if signal.valence in {"negative", "mixed"} and signal.intensity >= 0.45:
            grouped.setdefault(_pain_area(signal.label, product_area), []).append(signal)

    pain_points: list[PainPoint] = []
    for index, (area, signals) in enumerate(grouped.items(), start=1):
        rule = PAIN_POINT_RULES[area]
        evidence: list[Evidence] = []
        for signal in signals[:3]:
            evidence.extend(signal.evidence[:2])
        severity = "high" if mean([s.intensity for s in signals]) >= 0.7 else "medium"
        pain_points.append(
            PainPoint(
                pain_point_id=f"pp_{index:03d}",
                title=rule["title"],
                description=rule["description"],
                affected_user_segments=[interview.participant.get("segment", "sample_user")],
                journey_stage=area,
                product_area=area,
                severity=severity,
                frequency_note=f"Observed in {len(signals)} segment(s) in this sample interview.",
                evidence=evidence,
                emotion_signals=[s.label for s in signals],
                recommended_next_step=rule["recommended_next_step"],
            )
        )
    return pain_points


def generate_opportunities(pain_points: list[PainPoint]) -> list[ProductOpportunity]:
    opportunities: list[ProductOpportunity] = []
    for index, pain in enumerate(pain_points, start=1):
        opportunities.append(
            ProductOpportunity(
                opportunity_id=f"opp_{index:03d}",
                title=f"Reduce friction in {pain.product_area} / 降低 {pain.product_area} 环节摩擦",
                description=f"Turn the evidence behind '{pain.title}' into an MVP improvement experiment. 将该痛点背后的证据转化为一个 MVP 改进实验。",
                suggested_action=pain.recommended_next_step,
                success_metric="Increase evidence-backed task confidence and reduce repeated clarification needs. 提升有证据支撑的任务确认感，并减少重复澄清需求。",
                supporting_pain_points=[pain.pain_point_id],
                confidence=0.72 if pain.severity == "medium" else 0.82,
            )
        )
    return opportunities


def generate_structured_rationales(interview: Interview) -> list[StructuredRationale]:
    timeline = build_emotion_timeline(interview)
    top_signals = sorted(timeline, key=lambda item: item.intensity, reverse=True)[:3]
    rationales: list[StructuredRationale] = []
    for signal in top_signals:
        evidence_lines = [
            f"{ev.timestamp_start}-{ev.timestamp_end} [{ev.source}]: {ev.quote}"
            for ev in signal.evidence[:3]
        ]
        rationales.append(
            StructuredRationale(
                recognition=f"Detected an open-vocabulary '{signal.label}' expression signal with intensity {signal.intensity}.",
                evidence_attribution=evidence_lines,
                product_suggestion=(
                    "Review this moment as a potential product friction point and decide whether the flow needs clearer guidance, "
                    "a confirmation state, or follow-up research. 可将该片段视为潜在产品摩擦点，进一步判断是否需要更清晰的引导、确认态或后续研究。"
                ),
                safety_note=(
                    "This is a product research signal from mock metadata and transcript evidence. "
                    "It is not a psychological diagnosis and should not be used for high-stakes decisions. "
                    "这是基于 mock metadata 和访谈证据的产品研究信号，不是心理诊断，也不应用于高风险决策。"
                ),
            )
        )
    return rationales


def summarize_metrics(interview: Interview, pain_points: list[PainPoint]) -> dict[str, object]:
    timeline = build_emotion_timeline(interview)
    evidence_count = sum(len(point.evidence) for point in pain_points)
    label_counts = Counter(signal.label for signal in timeline)
    grounded_points = sum(1 for point in pain_points if point.evidence)
    return {
        "activation": "Sample loaded -> insights generated -> evidence reviewed",
        "time_to_insight": "< 5 seconds in local mock demo / 本地 mock demo 中少于 5 秒",
        "evidence_coverage": round(grounded_points / max(len(pain_points), 1), 2),
        "insight_usefulness": "Demo proxy: every opportunity includes an action and success metric / 每条机会点都包含行动建议和成功指标",
        "evidence_grounding": f"{evidence_count} evidence snippets linked to pain points",
        "emotion_consistency": dict(label_counts),
        "actionability": f"{len(pain_points)} pain point(s) mapped to product opportunities / {len(pain_points)} 个痛点已映射到产品机会",
    }


def generate_report(interview: Interview) -> InsightReport:
    pain_points = generate_pain_points(interview)
    opportunities = generate_opportunities(pain_points)
    rationales = generate_structured_rationales(interview)
    return InsightReport(
        interview_id=interview.interview_id,
        top_pain_points=pain_points,
        opportunities=opportunities,
        rationales=rationales,
        follow_up_questions=[
            "Which moments made users hesitate before trusting the product result? 哪些片段让用户在信任产品结果前产生犹豫？",
            "What confirmation or explanation would help users feel safe moving forward? 什么确认信息或解释能帮助用户更安心地继续？",
            "Which suggested action should be tested first in a usability study? 哪个产品改进动作最适合优先进入可用性测试？",
        ],
        metrics=summarize_metrics(interview, pain_points),
    )
