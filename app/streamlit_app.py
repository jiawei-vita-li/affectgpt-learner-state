from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.emotion_timeline import build_emotion_timeline
from core.insight_generator import generate_report
from core.media_analyzer import analyze_uploaded_media
from core.schemas import parse_interview


DATA_PATH = PROJECT_ROOT / "data" / "sample_interviews.json"


TEXT = {
    "zh": {
        "page_caption": "教育视频学习者状态反馈：engagement / confusion / frustration / boredom 时间线与多模态证据。",
        "safety": "本 demo 不用于学生打分或心理诊断；上传模式可调用面部表情模型，输出仅供教学产品优化参考。",
        "what_is_this": "这个网页在分析什么？",
        "what_is_this_body": (
            "AffectGPT Learner State 分析教学场景中的学习者状态。"
            "样例模式读取 `data/sample_interviews.json` 中的 MOOC/访谈式片段（含 mock 多模态 cues）；"
            "上传模式可对图片/视频做表情分类并生成状态时间线与结构化建议（课程优化 / 教师信号 / 自适应学习）。"
        ),
        "data_source": "原始数据来源",
        "data_source_body": "`data/sample_interviews.json`，包含 2 个 sample interviews。每条 segment 是一个“访谈片段”，不是一个真实视频文件。",
        "current_interview": "当前分析的访谈",
        "interview_context": "访谈上下文",
        "clip_note": (
            "- 页面把每条 transcript segment 当作一个 mock interview clip。\n"
            "- Mock cues 模拟真实多模态 pipeline 可能输出的特征。\n"
            "- 当前 repo 中没有真实视频/音频文件。"
        ),
        "controls": "Demo 控制台",
        "language": "界面语言",
        "choose_interview": "选择样例访谈",
        "input_mode": "输入方式",
        "sample_mode": "使用内置样例",
        "upload_mode": "上传真实文件",
        "upload_file": "上传文本 / 图片 / 视频",
        "upload_help": "支持 .txt/.json/.md/.png/.jpg/.mp4/.mov。图片/视频会用本地 Hugging Face 表情分类模型生成 facial expression signal。",
        "upload_notes": "访谈转写 / 观察笔记",
        "upload_notes_help": "可选。粘贴转写、用户原话或观察笔记后，系统会把表情模型结果和文本证据一起用于分析。",
        "uploaded_asset": "上传的原始素材",
        "uploaded_warning": "当前版本会自动对图片/视频抽帧做面部表情分类；该结果仅代表可见表情信号，不等同于心理诊断或真实内心状态。",
        "media_model_result": "图片/视频自动表情识别结果",
        "mvp_scope": "MVP 范围",
        "scope_items": "- 访谈文本浏览\n- 情绪时间线\n- 证据片段\n- 用户痛点\n- 产品机会",
        "segments": "分析片段数",
        "pain_points": "痛点数",
        "opportunities": "机会点",
        "evidence_coverage": "证据覆盖率",
        "tabs": ["摘要", "情绪时间线", "洞察", "Structured Rationale", "样例数据"],
        "research_goals": "研究目标",
        "top_pain_points": "Top 用户痛点",
        "suggested_actions": "建议产品动作",
        "followup_questions": "后续研究问题",
        "timeline_title": "开放词汇情绪时间线",
        "segments_and_cues": "访谈片段与 mock cues",
        "emotion_intensity": "情绪强度",
        "confidence": "置信度",
        "valence": "情绪倾向",
        "mock_cues": "Mock cues",
        "pain_cards": "痛点卡片",
        "opportunity_cards": "产品机会卡片",
        "severity": "严重度",
        "journey_stage": "用户旅程阶段",
        "recommended_next": "建议下一步",
        "evidence_snippets": "证据片段",
        "cue": "Cue",
        "suggested_action": "建议动作",
        "success_metric": "成功指标",
        "confidence_proxy": "置信度 proxy",
        "rationale": "Structured rationale",
        "rationale_chain": "**Recognition -> Evidence Attribution -> Product Suggestion**",
        "evidence_attribution": "**Evidence Attribution / 证据归因**",
        "product_suggestion": "**Product Suggestion / 产品建议**",
        "metrics": "产品与质量指标",
        "raw_data": "原始样例访谈",
        "download": "下载生成报告 JSON",
    },
    "en": {
        "page_caption": "Evidence-grounded multimodal interview insights for PMs and UX researchers.",
        "safety": "This demo can run local open-source facial expression classification on uploaded images/videos. It does not perform psychological diagnosis, clinical assessment, or real mental-state inference.",
        "what_is_this": "What is this webpage analyzing?",
        "what_is_this_body": (
            "AffectInsight is a user interview analysis demo. It does not play or analyze real video files. "
            "It reads sample interview segments from `data/sample_interviews.json`: each segment contains transcript text, timestamps, task context, "
            "and simulated audio / video / facial / text cues. The page shows how a PM can turn interview evidence into an emotion timeline, pain points, and product opportunities."
        ),
        "data_source": "Raw data source",
        "data_source_body": "`data/sample_interviews.json`, containing 2 sample interviews. Each segment is an interview moment, not a real video file.",
        "current_interview": "Current interview",
        "interview_context": "Interview context",
        "clip_note": (
            "- Each transcript segment is treated as a mock interview clip.\n"
            "- Mock cues simulate what a real multimodal pipeline might output.\n"
            "- There is no raw video/audio file in this repo."
        ),
        "controls": "Demo Controls",
        "language": "Interface language",
        "choose_interview": "Choose sample interview",
        "input_mode": "Input mode",
        "sample_mode": "Use built-in sample",
        "upload_mode": "Upload real file",
        "upload_file": "Upload text / image / video",
        "upload_help": "Supports .txt/.json/.md/.png/.jpg/.mp4/.mov. Images/videos are analyzed with a local Hugging Face facial expression classifier.",
        "upload_notes": "Interview transcript / observation notes",
        "upload_notes_help": "Optional. Add transcript, user quotes, or observation notes so the app can combine expression-model output with textual evidence.",
        "uploaded_asset": "Uploaded raw asset",
        "uploaded_warning": "This version automatically samples images/videos for facial expression classification. The result is a visible-expression signal, not a diagnosis or ground truth mental state.",
        "media_model_result": "Automatic image/video expression result",
        "mvp_scope": "MVP scope",
        "scope_items": "- Transcript review\n- Emotion timeline\n- Evidence snippets\n- Pain points\n- Product opportunities",
        "segments": "Segments analyzed",
        "pain_points": "Pain points",
        "opportunities": "Opportunities",
        "evidence_coverage": "Evidence coverage",
        "tabs": ["Summary", "Emotion Timeline", "Insights", "Structured Rationale", "Sample Data"],
        "research_goals": "Research goals",
        "top_pain_points": "Top user pain points",
        "suggested_actions": "Suggested product actions",
        "followup_questions": "Follow-up research questions",
        "timeline_title": "Open-vocabulary emotion timeline",
        "segments_and_cues": "Transcript segments and mock cues",
        "emotion_intensity": "Emotion intensity",
        "confidence": "Confidence",
        "valence": "Valence",
        "mock_cues": "Mock cues",
        "pain_cards": "Pain point cards",
        "opportunity_cards": "Product opportunity cards",
        "severity": "Severity",
        "journey_stage": "Journey stage",
        "recommended_next": "Recommended next step",
        "evidence_snippets": "Evidence snippets",
        "cue": "Cue",
        "suggested_action": "Suggested action",
        "success_metric": "Success metric",
        "confidence_proxy": "Confidence proxy",
        "rationale": "Structured rationale",
        "rationale_chain": "**Recognition -> Evidence Attribution -> Product Suggestion**",
        "evidence_attribution": "**Evidence Attribution**",
        "product_suggestion": "**Product Suggestion**",
        "metrics": "Product and quality metrics",
        "raw_data": "Raw sample interview",
        "download": "Download generated report JSON",
    },
}


@st.cache_data
def load_interviews() -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def emotion_dataframe(timeline) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment": signal.segment_id,
                "time": signal.timestamp_start,
                "emotion": signal.label,
                "intensity": signal.intensity,
                "confidence": signal.confidence,
                "valence": signal.valence,
            }
            for signal in timeline
        ]
    )


def split_text_into_segments(text: str) -> list[str]:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
    if len(lines) >= 2:
        return lines[:12]
    sentences = [item.strip() for item in re.split(r"[。！？!?\.]+", text) if item.strip()]
    return sentences[:12] if sentences else ["No transcript provided."]


def build_uploaded_interview(uploaded_file, notes: str, media_analysis: dict | None = None) -> dict:
    file_name = uploaded_file.name if uploaded_file else "manual_notes.txt"
    text = notes.strip()
    if uploaded_file and uploaded_file.type.startswith("text"):
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore").strip() or text
    elif uploaded_file and file_name.lower().endswith((".txt", ".md")):
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore").strip() or text
    elif uploaded_file and file_name.lower().endswith(".json"):
        try:
            raw = json.loads(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
            if isinstance(raw, dict) and "transcript_segments" in raw:
                return raw
            text = json.dumps(raw, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            text = notes.strip()

    if not text and media_analysis:
        expression = media_analysis.get("facial_expression", {})
        label = expression.get("label", "unknown")
        confidence = expression.get("confidence", 0.0)
        text = (
            f"Facial expression model detected '{label}' with confidence {confidence}. "
            "No transcript or observation note was provided."
        )

    segments = split_text_into_segments(text)
    transcript_segments = []
    for index, segment_text in enumerate(segments, start=1):
        normalized_text = segment_text.lower()
        uncertainty_words = sum(word in normalized_text for word in ["not sure", "maybe", "confused", "unclear", "hesitate", "不确定", "可能", "困惑", "不清楚", "犹豫"])
        negative_words = sum(word in normalized_text for word in ["frustrated", "stuck", "hard", "overwhelmed", "annoying", "挫败", "卡住", "困难", "复杂", "崩溃", "烦"])
        positive_words = sum(word in normalized_text for word in ["relieved", "clear", "helpful", "great", "works", "放松", "清楚", "有帮助", "不错", "能用"])
        cues = {
            "audio": {"pause_ratio": min(0.45, 0.12 + 0.08 * uncertainty_words), "pitch_delta": min(0.7, 0.2 + 0.12 * negative_words)},
            "facial": {"brow_tension": min(0.85, 0.25 + 0.18 * negative_words), "smile_score": min(0.8, 0.15 + 0.2 * positive_words), "eye_contact_score": 0.55},
            "video": {"gaze_shift_count": 2 + uncertainty_words + negative_words, "movement_energy": min(0.75, 0.25 + 0.12 * negative_words)},
            "text": {"sentiment": max(-0.8, min(0.8, 0.25 * positive_words - 0.25 * negative_words - 0.1 * uncertainty_words)), "uncertainty_words": uncertainty_words, "relief_words": positive_words},
        }
        if media_analysis and index == 1:
            cues["media_model"] = media_analysis

        transcript_segments.append(
            {
                "segment_id": f"upload_seg_{index:03d}",
                "timestamp_start": f"00:{index - 1:02d}:00",
                "timestamp_end": f"00:{index:02d}:00",
                "speaker": "participant",
                "text": segment_text,
                "context": {
                    "task": "uploaded interview analysis",
                    "screen": "uploaded_asset_review",
                    "prompt": "User-provided transcript or observation note",
                    "product_area": "onboarding" if index % 2 else "collaboration",
                },
                "cues": cues,
            }
        )

    return {
        "interview_id": "uploaded_interview",
        "project_id": "affectinsight_uploaded_analysis",
        "title": f"Uploaded analysis: {file_name}",
        "participant": {
            "participant_id": "uploaded_participant",
            "segment": "uploaded_user",
            "role": "unknown",
            "experience_level": "unknown",
            "metadata": {"source_file": file_name},
        },
        "session": {
            "date": "uploaded",
            "duration_minutes": len(transcript_segments),
            "interview_type": "uploaded transcript or observation notes",
            "consent_status": "user_provided_local_file",
            "recording_available": bool(uploaded_file),
            "transcript_available": bool(text),
        },
        "research_goals": [
            "Analyze user-provided transcript or notes for product friction signals.",
            "Use a local open-source facial expression classifier for uploaded images/videos when available.",
            "Generate evidence-grounded pain points and product opportunities.",
        ],
        "transcript_segments": transcript_segments,
    }


def tr(lang: str, key: str):
    return TEXT[lang][key]


def render_transcript(interview, timeline, lang: str) -> None:
    signal_by_segment = {signal.segment_id: signal for signal in timeline}
    for segment in interview.transcript_segments:
        signal = signal_by_segment[segment.segment_id]
        with st.expander(
            f"{segment.timestamp_start}-{segment.timestamp_end} | {signal.label} | {segment.context.get('screen', 'unknown screen')}",
            expanded=False,
        ):
            st.write(segment.text)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(tr(lang, "emotion_intensity"), f"{signal.intensity:.2f}")
            with col2:
                st.metric(tr(lang, "confidence"), f"{signal.confidence:.2f}")
            with col3:
                st.metric(tr(lang, "valence"), signal.valence)
            st.caption(tr(lang, "mock_cues"))
            st.json(segment.cues)


def render_pain_points(report, lang: str) -> None:
    for pain in report.top_pain_points:
        with st.container(border=True):
            st.subheader(pain.title)
            st.write(pain.description)
            st.write(f"**{tr(lang, 'severity')}:** {pain.severity}  |  **{tr(lang, 'journey_stage')}:** {pain.journey_stage}")
            st.write(f"**{tr(lang, 'recommended_next')}:** {pain.recommended_next_step}")
            st.write(f"**{tr(lang, 'evidence_snippets')}**")
            for evidence in pain.evidence:
                st.markdown(
                    f"- `{evidence.timestamp_start}` **{evidence.source}**: {evidence.quote}  \n"
                    f"  _{tr(lang, 'cue')}:_ {evidence.cue}"
                )


def render_opportunities(report, lang: str) -> None:
    for opportunity in report.opportunities:
        with st.container(border=True):
            st.subheader(opportunity.title)
            st.write(opportunity.description)
            st.write(f"**{tr(lang, 'suggested_action')}:** {opportunity.suggested_action}")
            st.write(f"**{tr(lang, 'success_metric')}:** {opportunity.success_metric}")
            st.progress(opportunity.confidence, text=f"{tr(lang, 'confidence_proxy')}: {opportunity.confidence:.0%}")


def render_rationales(report, lang: str) -> None:
    for rationale in report.rationales:
        with st.expander(rationale.recognition, expanded=False):
            st.write(tr(lang, "evidence_attribution"))
            for item in rationale.evidence_attribution:
                st.markdown(f"- {item}")
            st.write(tr(lang, "product_suggestion"))
            st.write(rationale.product_suggestion)
            st.warning(rationale.safety_note)


def main() -> None:
    st.set_page_config(page_title="AffectInsight", layout="wide")
    st.title("AffectInsight")
    with st.sidebar:
        language_label = st.radio("界面语言 / Interface language", ["中文", "English"], index=0)
    lang = "zh" if language_label == "中文" else "en"
    st.caption(tr(lang, "page_caption"))

    st.info(tr(lang, "safety"))
    with st.expander(tr(lang, "what_is_this"), expanded=True):
        st.write(tr(lang, "what_is_this_body"))
        st.write(f"**{tr(lang, 'data_source')}**")
        st.markdown(tr(lang, "data_source_body"))

    with st.sidebar:
        st.header(tr(lang, "controls"))
        input_mode = st.radio(tr(lang, "input_mode"), [tr(lang, "sample_mode"), tr(lang, "upload_mode")], index=0)
        st.write(f"**{tr(lang, 'mvp_scope')}**")
        st.markdown(tr(lang, "scope_items"))

    raw_interviews = load_interviews()
    interview_options = {item["title"]: item for item in raw_interviews}
    uploaded_file = None
    uploaded_notes = ""
    raw_selected_interview = None
    media_analysis = None

    if input_mode == tr(lang, "sample_mode"):
        with st.sidebar:
            selected_title = st.selectbox(tr(lang, "choose_interview"), list(interview_options.keys()))
        raw_selected_interview = interview_options[selected_title]
    else:
        uploaded_file = st.file_uploader(
            tr(lang, "upload_file"),
            type=["txt", "md", "json", "png", "jpg", "jpeg", "mp4", "mov"],
            help=tr(lang, "upload_help"),
        )
        uploaded_notes = st.text_area(
            tr(lang, "upload_notes"),
            value="I was confused during the setup screen.\nThere were too many options and I felt overwhelmed.\nSeeing the checklist made me feel relieved.",
            height=160,
            help=tr(lang, "upload_notes_help"),
        )
        st.warning(tr(lang, "uploaded_warning"))
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            st.subheader(tr(lang, "uploaded_asset"))
            if uploaded_file.type.startswith("image"):
                st.image(uploaded_file, use_container_width=True)
            elif uploaded_file.type.startswith("video"):
                st.video(uploaded_file)
            else:
                st.code(file_bytes.decode("utf-8", errors="ignore")[:4000])

            if uploaded_file.type.startswith(("image", "video")):
                with st.spinner(tr(lang, "media_model_result")):
                    try:
                        media_analysis = analyze_uploaded_media(uploaded_file.name, uploaded_file.type, file_bytes)
                    except Exception as error:  # noqa: BLE001 - show model-loading failure in the demo UI.
                        st.error(f"Model loading failed: {error}")
                        media_analysis = None
                if media_analysis:
                    st.subheader(tr(lang, "media_model_result"))
                    st.json(media_analysis)
        raw_selected_interview = build_uploaded_interview(uploaded_file, uploaded_notes, media_analysis)

    interview = parse_interview(raw_selected_interview)
    timeline = build_emotion_timeline(interview)
    report = generate_report(interview)
    timeline_df = emotion_dataframe(timeline)

    with st.container(border=True):
        st.subheader(tr(lang, "current_interview"))
        st.write(f"**{interview.title}**")
        st.write(f"Interview ID: `{interview.interview_id}` | Project ID: `{interview.project_id}`")
        st.write(f"Participant segment: `{interview.participant.get('segment', 'unknown')}` | Role: `{interview.participant.get('role', 'unknown')}`")
        st.write(f"**{tr(lang, 'interview_context')}**")
        st.markdown(tr(lang, "clip_note"))

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric(tr(lang, "segments"), len(interview.transcript_segments))
    summary_col2.metric(tr(lang, "pain_points"), len(report.top_pain_points))
    summary_col3.metric(tr(lang, "opportunities"), len(report.opportunities))
    summary_col4.metric(tr(lang, "evidence_coverage"), report.metrics["evidence_coverage"])

    tab_overview, tab_timeline, tab_insights, tab_rationale, tab_data = st.tabs(
        tr(lang, "tabs")
    )

    with tab_overview:
        st.subheader(interview.title)
        st.write(f"**{tr(lang, 'research_goals')}**")
        for goal in interview.research_goals:
            st.markdown(f"- {goal}")
        st.write(f"**{tr(lang, 'top_pain_points')}**")
        for pain in report.top_pain_points:
            st.markdown(f"- **{pain.title}**: {pain.description}")
        st.write(f"**{tr(lang, 'suggested_actions')}**")
        for opportunity in report.opportunities:
            st.markdown(f"- **{opportunity.title}**: {opportunity.suggested_action}")
        st.write(f"**{tr(lang, 'followup_questions')}**")
        for question in report.follow_up_questions:
            st.markdown(f"- {question}")

    with tab_timeline:
        st.subheader(tr(lang, "timeline_title"))
        st.line_chart(timeline_df.set_index("segment")[["intensity", "confidence"]])
        st.dataframe(timeline_df, use_container_width=True)
        st.subheader(tr(lang, "segments_and_cues"))
        render_transcript(interview, timeline, lang)

    with tab_insights:
        st.subheader(tr(lang, "pain_cards"))
        render_pain_points(report, lang)
        st.subheader(tr(lang, "opportunity_cards"))
        render_opportunities(report, lang)

    with tab_rationale:
        st.subheader(tr(lang, "rationale"))
        st.markdown(tr(lang, "rationale_chain"))
        render_rationales(report, lang)
        st.subheader(tr(lang, "metrics"))
        st.json(report.metrics)

    with tab_data:
        st.subheader(tr(lang, "raw_data"))
        st.json(raw_selected_interview)
        st.download_button(
            tr(lang, "download"),
            data=json.dumps(asdict(report), ensure_ascii=False, indent=2),
            file_name=f"{interview.interview_id}_affectinsight_report.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
