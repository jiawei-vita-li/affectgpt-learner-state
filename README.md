# AffectGPT — Learner State Feedback for E-Learning Video

> Instructional video / camera feed / sample interview → multimodal **learner-state timeline** with evidence + three downstream signal cards (course design / live-class teacher / adaptive engine).
>
> 输入教学视频、摄像头画面或样例访谈 → 输出带证据支撑的**学习者状态时间线**，并生成三类下游信号（课程优化、直播教学、自适应引擎）。

**Hard boundary**: this is a learning-experience product. It is **not for** student grading, hiring or admissions filtering, exam scoring, or clinical psychological diagnosis.

**使用边界**：面向在线学习体验分析与教学改进，**不用于**学生打分、招聘或招生筛选、考试评分，也不做临床心理诊断。

---

## What you get

| Layer | Content |
| --- | --- |
| **Algorithm** | 5-label open-vocabulary emotion set · multi-source weighted cue fusion · HuggingFace `trpakov/vit-face-expression` for facial-expression classification (top-3) · OpenCV low-level visual metrics + Haar face count |
| **Product** | Bilingual Streamlit (ZH/EN) · sample-interview replay + image/video upload · pain points · product opportunities · structured rationale with mandatory `safety_note` · grounding metrics surfaced as a dict |
| **Engineering** | 5 unittest cases validating the schema contract, evidence grounding, and safety-note string · dataclass-typed schemas · explicit `source` whitelist per evidence atom |

| 层次 | 内容 |
| --- | --- |
| **算法** | 五标签开放词表情绪集 · 多源加权 cue 融合 · HuggingFace 表情分类（top-3）· OpenCV 低层视觉指标 + Haar 人脸检测 |
| **产品** | 中英双语 Streamlit · 样例访谈回放 + 图片/视频上传 · 学习痛点 · 产品机会 · 结构化 rationale（含必填 `safety_note`）· 证据 grounding 指标 |
| **工程** | 5 个 unittest 校验 schema、证据绑定与安全说明 · dataclass 类型约束 · 每条证据标注 `source` 白名单 |

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The first image upload triggers a ~90 MB HuggingFace model download.

首次上传图片时会下载约 90 MB 的 HuggingFace 模型（仅首次）。

### Run tests

```bash
python -m unittest discover -s tests
```

5 tests pin: schema parsing + source whitelist, ≥ 2 distinct emotion labels per interview with evidence on every signal, top-N pain points all grounded, mandatory `safety_note` string equals `"this is not a psychological diagnosis"`, and the 7-key metrics dict contract.

5 个测试校验：schema 与 source 白名单、每条访谈至少 2 类情绪标签且均有证据、Top-N 痛点均有片段依据、`safety_note` 含「这不是心理诊断」、7 项 metrics 字段契约。

---

## Documentation

- **[docs/PRD_full.md](docs/PRD_full.md)** — Full bilingual PRD (17 sections) / 完整双语 PRD（17 章）
- **[docs/TECH_REPORT.md](docs/TECH_REPORT.md)** — Technical report + verified results + Phase 2 plan / 技术报告、已验证成果与 Phase 2 计划
- [docs/PRD.md](docs/PRD.md) — TL;DR / 概要
- [docs/metrics.md](docs/metrics.md) · [docs/competitive_analysis.md](docs/competitive_analysis.md) · [docs/user_research.md](docs/user_research.md) · [docs/roadmap.md](docs/roadmap.md) · [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md)
- [outputs/EVAL_RESULTS.md](outputs/EVAL_RESULTS.md) — DAiSEE evaluation (Phase 2 fill-in template) / DAiSEE 评估（Phase 2 待填写模板）

---

## Phase split

The repo clearly separates what is verified today from what is planned.

本仓库明确区分「当前已验证」与「后续计划」。

| Phase | What runs today | Notes |
| --- | --- | --- |
| **Phase 1** (this repo) | Text + low-level visual + HuggingFace face-expression classifier · 5 open-vocab labels mapped to canonical 4-class (DAiSEE-aligned) · structured rationale with safety_note · 5 unittest cases | Every evidence atom names its `source` so reviewers cannot mistake placeholder cues for real model output |
| **Phase 2** (designed) | MediaPipe face landmarks · ViViT / VideoMAE visual encoder · WavLM audio · Whisper ASR · sentence-transformers open-vocab retrieval · Qwen2.5-7B for rationale only (never classification) · DAiSEE 4-class weighted F1 + per-class F1 · FastAPI `POST /analyze_learner_state` + agent tool schema | To be added once Phase 2 runs |

| 阶段 | 当前可运行 | 说明 |
| --- | --- | --- |
| **Phase 1**（本仓库） | 文本 + 低层视觉 + HuggingFace 表情分类 · 五标签映射为 DAiSEE 对齐的四类 · 结构化 rationale 含 safety_note · 5 个 unittest | 每条证据标注 `source`，避免把占位 cue 误认为真实模型输出 |
| **Phase 2**（已设计） | MediaPipe 人脸关键点 · ViViT/VideoMAE · WavLM · Whisper ASR · 开放词表检索 · Qwen2.5-7B 仅生成 rationale（不参与分类）· DAiSEE 加权 F1 · FastAPI + Agent tool schema | Phase 2 跑通后补充 |

Phase 2 numbers not yet executed are labeled `TBD` in `outputs/EVAL_RESULTS.md`.

尚未执行的 Phase 2 指标均在 `outputs/EVAL_RESULTS.md` 中标注为 `TBD`。

---

## Responsible AI

See [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md). Key commitments:

完整说明见 [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md)。核心承诺如下：

- **Not for** punitive analytics, grading, hiring, exam scoring, or clinical diagnosis.  
  **不用于**惩罚性分析、学业打分、招聘筛选、考试评分或临床诊断。
- **Mandatory `safety_note`** — `StructuredRationale.safety_note` is a required field; a unit test enforces the disclaimer string.  
  **安全说明必填** — `safety_note` 为必填字段，单元测试锁定免责文案。
- **Low-confidence routes to human review**, never auto-notifies the student or teacher.  
  **低置信结果走人工复核**，不自动向学生或教师推送通知。
- **No raw-video retention** in the demo path.  
  **demo 路径不保留原始视频**。
- LLM (planned Qwen2.5-7B) writes rationale only; classification stays with encoders + fusion head—hallucinations cannot poison product labels.  
  大模型（计划 Qwen2.5-7B）仅撰写说明文字，分类由编码器与融合头完成，避免幻觉污染状态标签。

---

## License

MIT — see [LICENSE](LICENSE).
