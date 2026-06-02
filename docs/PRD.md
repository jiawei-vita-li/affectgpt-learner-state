# PRD: AffectGPT — Learner State Feedback

> TL;DR entry. For the full bilingual 17-section PRD see [`PRD_full.md`](PRD_full.md).
> 简短入口。完整双语 17 章 PRD 见 [`PRD_full.md`](PRD_full.md)。

## One-liner / 一句话

Instructional video → learner-state timeline → multimodal evidence → three downstream signal cards for course designers, live-class teachers, and adaptive-learning engines.
教学视频 → 学习者状态时间线 → 多模态证据 → 三类下游信号卡（课程设计、直播教师、自适应学习引擎）。

## Not in scope / 明确非目标

Student grading. Hiring or admissions filtering. Exam scoring. Clinical mental-health diagnosis. Generic UX research pain-point mining.
学生打分、招聘/招生筛选、考试评分、临床心理诊断、通用 UX 研究痛点挖掘。

## MVP (this repo) / MVP（本仓库）

- Streamlit: sample-interview replay + image / video upload / 样例回放 + 图片视频上传
- 5 open-vocabulary emotion labels mapped to canonical 4-class (DAiSEE-aligned) / 5 开放词汇 → 规范 4 类
- HuggingFace `trpakov/vit-face-expression` for facial-expression classification / HuggingFace 面部表情分类
- OpenCV + Haar for low-level visual + face count / OpenCV + Haar
- Mandatory `safety_note = "this is not a psychological diagnosis"` enforced by tests / `safety_note` 强制由测试钉住

## Phase 2 (designed)

MediaPipe · ViViT / VideoMAE · WavLM · Whisper · sentence-transformers · Qwen2.5-7B (rationale only) · DAiSEE 4-class weighted F1 + per-class F1 · FastAPI + agent tool schema → `outputs/EVAL_RESULTS.md`.

See [`TECH_REPORT.md`](TECH_REPORT.md) for the engineering plan.
工程计划见 [`TECH_REPORT.md`](TECH_REPORT.md)。
