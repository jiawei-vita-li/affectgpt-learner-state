# PRD — AffectGPT: Learner State Feedback for E-Learning Video / 在线教育学习者状态反馈

| Field / 字段 | Value / 值 |
| --- | --- |
| Version / 版本 | v1.0 |
| Status / 状态 | Phase 1 reference implementation shipped (text + low-level visual + HF face-expression classifier) · Phase 2 (ViViT / WavLM / Whisper / Qwen + DAiSEE eval) planned / Phase 1 参考实现已上线（文本 + 低层视觉 + HuggingFace 面部表情分类）；Phase 2（ViViT / WavLM / Whisper / Qwen + DAiSEE 评估）计划中 |
| Last updated / 更新于 | 2026-05-28 |
| Related docs / 相关文档 | [PRD.md](PRD.md) (TL;DR) · [RESPONSIBLE_AI.md](RESPONSIBLE_AI.md) · [competitive_analysis.md](competitive_analysis.md) · [metrics.md](metrics.md) · [user_research.md](user_research.md) · [roadmap.md](roadmap.md) · [TECH_REPORT.md](TECH_REPORT.md) |

---

## 1. TL;DR / 概要

在线课只能看到完课率，看不到学习者在**哪一秒**走神、困惑或挫败。直播老师无法盯每张脸。自适应引擎需要结构化信号，而不是一个 emoji 标签。
Online courses measure completion rate but cannot tell *when* a learner gets confused, frustrated, or disengaged. Live-class teachers cannot watch every face. Adaptive-learning engines need structured signals, not a single emoji label.

**AffectGPT Learner State** 接受教学视频 / 摄像头流 / 样例访谈片段，输出**多模态学习者状态时间线**：四个规范标签（`engagement / confusion / frustration / boredom`，底层由 5 标签开放词汇集 `frustrated / confused / hesitant / relieved / overwhelmed` 支撑）+ 证据归因 + 三类下游信号卡（课程设计优化 / 直播教师 dashboard / 自适应触发）。
**AffectGPT Learner State** ingests instructional video / camera feed / sample interview segments and returns a **multimodal learner-state timeline** with four canonical labels (`engagement / confusion / frustration / boredom`, backed by 5 open-vocab labels), evidence attribution, and three downstream signal cards.

**硬边界**：本产品**不用于**学生打分、招生/招聘筛选、心理诊断或任何高风险个体判断。
**Hard boundary:** this product is **not for** student grading, hiring/admissions filtering, psychological diagnosis, or any high-stakes individual judgment.

---

## 2. Problem statement / 问题陈述

### 2.1 What current online courses can and cannot see / 现在的在线课能看到与看不到的

| Layer / 层 | Today / 现状 | Gap / 缺口 |
| --- | --- | --- |
| Completion rate / 完课率 | Visible / 可见 | Doesn't tell *why* learners drop off / 看不到学生为何流失 |
| Quiz score / 测验分数 | Visible / 可见 | After-the-fact / 事后指标 |
| Click-stream / 点击流 | Visible / 可见 | Behavioral proxy, ignores affect / 行为代理，忽略情感 |
| Per-segment affective state / 片段级情感状态 | Not visible / 不可见 | **The wedge / 这是我们的切口** |

### 2.2 Why generic emotion APIs miss / 通用情绪 API 为什么不行

- **Hume / Affectiva** 每张脸每帧一个标签 — 无片段级 rationale、无字幕 grounding、无产品 action 链接。Per-frame label only — no segment-level rationale, no transcript grounding, no link to product action.
- **Sentiment APIs**（仅文本）漏掉视觉困惑与音频犹豫。Text-only — misses visual confusion + audio hesitation.
- **Academic affect-recognition models** 准但作为研究代码发布，没有 Responsible AI 脚手架，也没有 PM 可用 dashboard。Accurate but ship as research code, no Responsible-AI scaffolding, no PM dashboard.

### 2.3 Hypothesis / 核心假设

一个 (a) 限定**教育**场景的垂直产品、(b) 输出**带证据的片段级状态时间线**、(c) 发出**面向三类角色的信号卡**而不是裸标签，会比通用情绪 API 对课程设计师 / 教师 / 自适应引擎更有用。
A vertical product that (a) restricts itself to the **education** scenario, (b) returns a **state timeline with segment-level evidence**, and (c) emits **three role-specific signal cards** rather than raw labels will be more useful than a generic emotion API.

---

## 3. Target users / 目标用户

| Persona / 用户画像 | Job to be done / 待办任务 | Success criterion / 成功标准 |
| --- | --- | --- |
| **Course design PM / 课程设计 PM** | Find the 3 segments most likely to lose learners / 找出最易流失学生的 3 个片段 | Top-3 confusion peaks with frame + transcript evidence / Top-3 困惑峰带帧+字幕证据 |
| **Live-class product PM / 直播课产品 PM** | Real-time engagement curve without manual face-watching / 不靠人工盯脸的实时投入曲线 | A dashboard signal that triggers teacher check-in on sustained frustration / 持续 frustration 时触发教师 check-in |
| **Adaptive-learning owner / 自适应学习负责人** | Structured event stream the LMS can consume / 给 LMS 用的结构化事件流 | JSON events `{timestamp, state, confidence}` posted to webhook / JSON 事件 post 到 webhook |

非目标用户（明确排除）：
Non-users (deliberately):

- HR / 招生团队找情感筛选器。HR / admissions teams looking for an affective filter.
- 无法保证同意与未成年保护的 K-12 学生监控。K-12 student-monitoring without consent guarantees.
- 临床心理诊断。Clinical mental-health diagnosis.

---

## 4. User stories / 用户故事

### 4.1 Course-design PM — "Module post-mortem" / 课程设计 PM — 模块复盘

> *作为课程设计 PM，每次模块发布后我希望上传录制视频就能立刻看到 confusion / frustration 聚集的 Top-3 片段，附帧画面 + 字幕引用，让我在下一期开班前重写这些片段。*
> *As a course-design PM, after each module release, I want to upload the recorded class video and instantly see the top-3 confusion / frustration clusters with frame stills + transcript quotes, so I can rewrite those segments before the next cohort starts.*

Acceptance / 验收：
- 上传 → 时间线 + 各规范状态强度柱。Upload → timeline with intensity bars per canonical state.
- 负面价高强度 Top-3 高亮。Top-3 high-intensity negative-valence segments highlighted.
- 每个高亮段展示：时间戳、字幕片段、视觉 cue 来源、模型 rationale。Each shows: timestamp, transcript snippet, visual cue source, model rationale.

### 4.2 Live-class PM — "Real-time check-in" / 直播课 PM — 实时关怀

> *作为直播课产品 PM，只有当 frustration 在多个片段持续时（不是单次皱眉）才发信号，避免老师 dashboard 变成通知刷屏。*
> *As a live-class product PM, I want a signal that fires only when frustration is sustained across multiple segments (not a single frowny moment), so the teacher dashboard doesn't become a notification spammer.*

Acceptance / 验收：
- 持续状态检测用连续片段窗口，不是单点。Sustained-state detection uses a window over consecutive segments.
- 信号携带 `confidence`；低置信导向"人工 review"而非"自动通知"。Signal carries `confidence`; low confidence routes to human-review copy, not auto-notify.

### 4.3 Adaptive-learning owner — "Trigger contract" / 自适应学习 — 触发契约

> *作为自适应引擎负责人，我希望一份稳定的 JSON 事件契约，让规则引擎可以决定"插入复习" / "降低难度" / "什么都不做"，不必每次重新解析自由格式模型输出。*
> *As the owner of an adaptive engine, I want a stable JSON event contract so my rule engine can decide between "insert review" / "lower difficulty" / "do nothing" without re-parsing free-form output.*

Acceptance / 验收：
- 状态事件 schema 文档化。Documented schema for state events.
- 事件包含规范 4 类 `state` 加底层开放词汇标签与 intensity。Event includes canonical `state` plus the underlying open-vocab label + intensity.

### 4.4 (Cross-cutting) — "Audit any signal" / 通用 — 审计任意信号

> *作为任一上述用户，当我质疑某个状态标签时，我希望一键看到具体证据（字幕引用、音频 cue、面部表情模型输出、低层视觉指标），并附"非心理诊断"声明。*
> *As any of the above users, when I question a state label, I want to click it and see the exact evidence (transcript quote, audio cue, facial expression model output, low-level visual metrics) plus the disclaimer that this is not a psychological diagnosis.*

Acceptance / 验收：
- 每条状态信号带 `evidence` 列表，每个 atom 含 `(source, quote, cue, confidence)`。Every signal carries an `evidence` list of `(source, quote, cue, confidence)` atoms.
- 结构化 rationale 面板含**强制 `safety_note` 字段**附诊断免责。Structured rationale panel includes the mandatory `safety_note` field.

---

## 5. Use scenarios / 使用场景

### 5.1 Scenario A — "Sample MOOC segment review" (Phase 1 shipped) / 场景 A — MOOC 样例片段复盘（已上线）

1. 课程设计 PM 打开 Streamlit，从 `data/sample_interviews.json` 选择 2 条样例之一。Course-design PM picks one of two sample interviews.
2. 应用构建情绪时间线（5 标签开放词汇）→ 推出 Top-3 痛点 → 推出产品机会 + 跟进问题 + 结构化 rationale。Pipeline produces timeline → pain points → opportunities → rationales.
3. PM 浏览痛点，逐条打开证据列表和推荐下一步。PM scans pain points; opens evidence and recommended next step.

### 5.2 Scenario B — "Upload image / video" (Phase 1 shipped) / 场景 B — 上传图片/视频（已上线）

1. 用户上传 `.png / .jpg / .mp4 / .mov`。User uploads.
2. `media_analyzer` 采样最多 8 帧，计算低层视觉指标（brightness / contrast / saturation / edge density / warmth），用 Haar cascade 数脸数，并用 HuggingFace `trpakov/vit-face-expression` 分类面部表情（top-3）。Samples up to 8 frames, computes metrics, counts faces, classifies expression.
3. 结果融入片段 cue；情绪时间线同时使用文本规则 + 模型表情标签。Results merge into segment cues.
4. UI 标注："面部表情分类 — 不是心理诊断"。UI labels: "facial-expression classification — not psychological diagnosis".

### 5.3 Scenario C — "Phase 2: real teaching video" (planned) / 场景 C — Phase 2 真实教学视频（计划）

1. 用户上传完整教学视频。User uploads a full instructional video.
2. Pipeline：MediaPipe 面部 + ViViT/VideoMAE 视觉 + WavLM 音频 + Whisper ASR → 开放词汇短语检索（sentence-transformers）→ Qwen2.5-7B-Instruct 生成片段级 rationale 模板。MediaPipe + ViViT + WavLM + Whisper → sentence-transformers → Qwen2.5-7B.
3. 输出：状态时间线、证据面板、三类下游信号卡、JSON 事件流。Outputs: timeline, evidence panel, three signal cards, JSON events.
4. 可选：FastAPI `POST /analyze_learner_state` 给 Agent / LMS 集成。Optional FastAPI endpoint for agent / LMS integration.

---

## 6. Product principles / 产品原则

1. **证据先于标签 / Evidence first, label second.** 无证据不发声。No claim ships without a clickable evidence trail.
2. **垂直优于横向 / Vertical over horizontal.** 只优化教育，不外溢到用户研究 / 临床 / 招聘 / 考核。Optimize for education; refuse generalization.
3. **任何高风险决策必须有人在环 / Human in the loop on every high-stakes decision.** 低置信走人工 review，不走自动通知。Low-confidence routes to human review, not auto-notify.
4. **两层模型分工 / Compose two model layers but separate their responsibilities.** Encoder/分类器做感知；LLM（Qwen）只写 rationale —— 不参与分类。Encoders classify; LLM only narrates.
5. **Responsible AI 由 schema 强制 / Responsible AI is enforced by schema.** `StructuredRationale.safety_note` 是必填字段；测试断言诊断免责必须存在。`safety_note` is required; tests pin its presence.

---

## 7. MVP scope / MVP 范围

### 7.1 In scope (Phase 1 — shipped) / Phase 1 已上线

- 样例访谈模式（2 访谈 × 4 segment）：文本规则 + cue 驱动情绪时间线。Sample interview mode with text rules + cue-driven timeline.
- 上传模式：图片/视频 → HuggingFace ViT 面部表情分类（`trpakov/vit-face-expression`）+ OpenCV 低层视觉 + Haar 脸数。Upload mode with HF ViT + OpenCV + Haar.
- 5 个开放词汇标签：`frustrated / confused / hesitant / relieved / overwhelmed`，各自映射 valence。Five open-vocab labels mapped to valence.
- 每片段 `EmotionSignal`：rationale + 多 evidence atom + 受限 `intensity` 与 `confidence`。Per-segment signal with bounded fields.
- 痛点生成、产品机会卡、含 safety note 的结构化 rationale、摘要指标 dict。Pain points + opportunities + rationales + metrics.
- Streamlit ZH/EN UI（默认中文）。Bilingual UI.
- 5 条 unittest 覆盖：parse 正确性、时间线标签多样性、证据 grounded 的痛点、safety note、metrics dict 契约。5 unittest cases.

### 7.2 In scope (Phase 2 — designed, GPU-ready) / Phase 2 已设计，GPU 环境就绪

- Pipeline：MediaPipe + ViViT / VideoMAE + WavLM + Whisper。Pipeline modules.
- 开放词汇短语检索（sentence-transformers）。Open-vocab retrieval.
- Qwen2.5-7B-Instruct 生成 rationale（模板化，不分类）。Qwen rationale (template, not classification).
- DAiSEE 4 类（`boredom / engagement / confusion / frustration`）weighted F1 + per-class F1。DAiSEE 4-class evaluation.
- 可选 Qwen2.5-VL zero-shot baseline（仅定性对比）。Optional Qwen2.5-VL zero-shot.
- FastAPI `POST /analyze_learner_state` + Agent 工具 schema。FastAPI endpoint + agent schema.

### 7.3 Out of scope (explicit non-goals) / 明确非目标

- 学生打分、GPA 预测、招生筛选。Student grading, GPA prediction, admissions filtering.
- 临床心理诊断。Clinical mental-health diagnosis.
- 多说话人课堂分离（延至 v2）。Multi-speaker diarization (v2).
- 招聘 / 职场监控。Hiring or workplace surveillance.
- 与 Hume / Affectiva 竞争通用情绪 API。General-purpose emotion API competing with Hume.

---

## 8. Functional requirements / 功能需求

| # | Requirement / 需求 | Priority / 优先级 | Phase |
| --- | --- | --- | --- |
| FR-1 | Sample-interview replay with timeline + pain points + opportunities + rationales / 样例回放含时间线/痛点/机会/rationale | P0 | 1 ✓ |
| FR-2 | Upload mode (image/video) with HF face-expression classifier / 上传图片视频走 HF ViT | P0 | 1 ✓ |
| FR-3 | 5-label open-vocab `EmotionSignal` with bounded intensity ∈ [0,1] / 5 标签 EmotionSignal，intensity ∈ [0,1] | P0 | 1 ✓ |
| FR-4 | Multi-source evidence per signal / 每信号多源证据 | P0 | 1 ✓ |
| FR-5 | Structured rationale ships with `safety_note` field / 结构化 rationale 必含 safety_note | P0 | 1 ✓ |
| FR-6 | Metrics dict exposes 7 keys / metrics dict 暴露 7 字段 | P0 | 1 ✓ |
| FR-7 | ZH/EN bilingual UI / 中英 UI | P2 | 1 ✓ |
| FR-8 | Multimodal pipeline (MediaPipe + ViViT + WavLM + Whisper) / 多模态 pipeline | P0 | 2 |
| FR-9 | DAiSEE 4-class weighted F1 + per-class F1 / DAiSEE F1 | P0 | 2 |
| FR-10 | Qwen2.5-7B rationale (rationale only) / Qwen 只写 rationale | P1 | 2 |
| FR-11 | FastAPI `POST /analyze_learner_state` | P1 | 2 |
| FR-12 | Agent tool schema documenting JSON event stream / Agent 工具 schema | P2 | 2 |
| FR-13 | Sustained-state detection window / 持续状态检测窗口 | P2 | 2 |
| FR-14 | Confidence threshold + human-review routing copy / 置信阈值 + 人工 review 路由文案 | P0 | 2 |

---

## 9. UX & interaction flow / 体验与交互

### 9.1 Layout (Phase 1) / 布局

```
┌──────────────────────────────────────────────────────────────────────┐
│ Sidebar: language · controls · interview chooser · input mode        │
├──────────────────────────────────────────────────────────────────────┤
│ Header: "学习者状态反馈" · safety banner ("not for grading")          │
│ Row 1: Current interview context (title, participant, goals)         │
│ Row 2: Emotion timeline (segment bars with labels)                   │
│ Row 3: Pain points (title · evidence list · recommended next step)   │
│ Row 4: Product opportunities (action · success metric · confidence)  │
│ Row 5: Structured rationales (recognition · evidence · suggestion ·  │
│         safety note)                                                  │
│ Row 6: Metrics summary (7 fields)                                    │
│         If upload mode: face count + model top-3 + visual metrics    │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 Trust micro-interactions / 信任微交互

- 顶部安全 banner："本 demo 不用于学生打分或心理诊断"。Top safety banner.
- 每条面部表情模型输出附"visible facial expression, not internal state"文案。Each facial-expression output carries a visibility disclaimer.
- 低置信（< 0.55）信号视觉降权并标 "review"。Low-confidence signals UI-demoted and labeled "review".
- safety_note 文案 "not a psychological diagnosis" 由 `test_structured_rationale_has_safety_boundary` 强制存在。Disclaimer asserted by test.

### 9.3 Phase 2 additions / Phase 2 增量

- 真视频上传 + 每秒时间线滑块。Real video upload with per-second scrubber.
- 三类信号卡（课程设计 / 直播 / 自适应）显式标注 "to whom"。Three signal cards with explicit audience tag.
- 自适应契约的 JSON 事件预览。JSON event preview.

---

## 10. Data model / 数据模型

Defined in `core/schemas.py`. 定义于 `core/schemas.py`。

| Type / 类型 | Key fields / 关键字段 | Purpose / 用途 |
| --- | --- | --- |
| `Evidence` | segment_id · quote · source · timestamps · cue · confidence | Per-cue evidence atom / 单 cue 证据 atom |
| `TranscriptSegment` | segment_id · timestamps · speaker · text · context · cues | One transcript slice / 一条 transcript 切片 |
| `Interview` | interview_id · project_id · title · participant · session · goals · segments | Full sample interview / 一份样例访谈 |
| `EmotionSignal` | segment_id · timestamps · label · canonical_label · valence · intensity · confidence · evidence · rationale | Affective inference for one segment / 单段情感推断 |
| `PainPoint` | id · title · description · affected_segments · journey_stage · product_area · severity · evidence · emotion_signals · recommended_next_step | Aggregated pain / 聚合痛点 |
| `ProductOpportunity` | id · title · description · suggested_action · success_metric · supporting_pain_points · confidence | Action card / 行动卡 |
| `StructuredRationale` | recognition · evidence_attribution · product_suggestion · **safety_note** | Auditable rationale / 可审计 rationale |
| `InsightReport` | interview_id · top_pain_points · opportunities · rationales · follow_up_questions · metrics | Top-level output / 顶层输出 |

Phase 2 将（不破坏地）扩展 schema：
Phase 2 extends (not breaks) the schema:

- `state ∈ {engagement, confusion, frustration, boredom}` 规范 4 类（对齐 DAiSEE）。Canonical 4-class.
- 来自 ASR 的真实时间戳。Real ASR timestamps.
- ViViT / WavLM / MediaPipe 的模型输出 cue 字段。Model-output cue fields.

---

## 11. Success metrics / 成功指标

### 11.1 North star / 北极星

**上传到第一个 actionable 片段标记的时间** — 10 分钟视频目标 < 5 分钟（Phase 2）。
**Time from upload to first actionable segment flagged** — target < 5 min on a 10-min video (Phase 2).
Phase 1 参考代理：本地 mock demo time-to-insight < 5 秒。Phase 1 proxy: local mock < 5 s.

### 11.2 Quality / 质量

| Metric / 指标 | Definition / 定义 | Phase 1 evidence / Phase 1 证据 | Phase 2 target / Phase 2 目标 |
| --- | --- | --- | --- |
| Evidence Coverage | `grounded_pain_points / total_pain_points` | computed in `summarize_metrics()` and surfaced in UI / 在指标摘要中计算并 UI 暴露 | ≥ 0.9 |
| Evidence Grounding | every signal has at least one evidence atom / 每信号至少 1 个 evidence | asserted in `test_open_vocabulary_emotion_timeline` / 测试断言 | maintained at 1.0 |
| DAiSEE weighted F1 | 4-class affective state / 4 类情感状态 | — | TBD |
| Per-class F1 | confusion / engagement / boredom / frustration | — | reported for diagnosis / 输出做诊断 |
| Confusion-moment segment recall | flagged segments vs human-marked confusion peaks / 标记片段 vs 人工标注峰 | — | TBD |
| Rationale faithfulness | does rationale reference the cited evidence / rationale 是否只引用真实证据 | — | TBD (human rubric) |

### 11.3 Trust / safety / 信任 + 安全

- 100% 信号携带 safety-note（测试断言）。100% of signals carry the safety-note string (test-asserted).
- 低置信信号永不自动发送给老师（Phase 2 UX 约束）。Low-confidence never auto-notify teachers (Phase 2 UX constraint).

### 11.4 Guardrails / 兜底

- Unsupported-diagnostic-language rate = 0（任意非零即回归）。Unsupported diagnostic language rate = 0.
- High-stakes-decision recommendation rate = 0（不出现"建议劝退"类文案）。No "drop this student" copy.
- Privacy-sensitive-data exposure rate = 0（Phase 2: demo 默认不留存原视频）。No raw-video retention by default.

完整定义与事件设计：`docs/metrics.md`。Full prose: `docs/metrics.md`.

---

## 12. Competitive analysis (summary) / 竞品分析（摘要）

| Product / category / 产品/品类 | Strength / 优势 | Gap vs us / 与我们的差距 |
| --- | --- | --- |
| Hume / Affectiva | Strong perceptual models / 感知模型强 | No segment-level rationale; not vertical to education / 无片段级 rationale，非教育垂直 |
| Sentiment APIs / 情感分析 API | Cheap, simple / 便宜简单 | Text only; rigid polarity / 仅文本，刚性极性 |
| Academic affect models (DAiSEE, AffectNet trained) / 学术情感模型 | Strong accuracy / 高精度 | Research code, no RAI scaffolding, no PM dashboard / 研究代码 |
| LMS analytics dashboards / LMS 分析 dashboard | Aggregate completion + quiz / 聚合完课与测验 | No per-segment affective state / 无片段级情感 |
| Open-source video classifiers / 开源视频分类器 | Generic / 通用 | No evidence panel, no agent contract / 无证据面板、无 Agent 契约 |

完整对比与差异化：`docs/competitive_analysis.md`。Full table & wedge: `docs/competitive_analysis.md`.

---

## 13. Responsible AI / 负责任 AI

详见 `docs/RESPONSIBLE_AI.md`。PRD 级承诺：
See `docs/RESPONSIBLE_AI.md`. PRD-level commitments:

- **不用于 / Not for**：惩罚性分析、招聘、考核、临床诊断。Punitive analytics, hiring, exam scoring, clinical diagnosis.
- **未成年人 / Minors**：需要 consent；demo 默认不留存原视频。Consent required; demo does not retain raw video by default.
- **教师在环 / Teachers in the loop**：信号辅助，不替代教师判断。Signals assist, not replace.
- **低置信 / Low confidence**：路由人工 review，不走自动学生侧干预。Routes to human review, never auto student-facing intervention.
- **safety_note 字段 / Safety-note field**：结构化必填；测试断言。Structurally required; test-pinned.
- **文化偏差 / Cultural bias**：面部表情与情感模型带文化偏差 — Phase 2 计划包含偏差感知 review 与显式 caveat 文案。Phase 2 plan includes bias-aware review + explicit caveat.

---

## 14. Risks & mitigations / 风险与缓解

| Risk / 风险 | Likelihood / 概率 | Impact / 影响 | Mitigation / 缓解 |
| --- | --- | --- | --- |
| Users over-trust emotion labels / 用户过度信任情感标签 | High / 高 | High / 高 | Safety banner + per-signal disclaimer + low-confidence demotion / 三层降权 |
| Mock metadata mistaken for real output / mock 元数据被误认为真模型输出 | Medium / 中 | Medium / 中 | Every evidence atom names its `source` (`transcript / mock_audio_metadata / mock_facial_metadata / facial_expression_model`) / 每 atom 标注 source |
| Cultural / demographic bias in affect models / 情感模型文化与人口偏差 | High / 高 | High / 高 | Phase 2 includes bias caveat + rubric; no individual auto-decision / Phase 2 加偏差 caveat 与 rubric |
| Use-case drift to grading / surveillance / 用例漂向打分/监控 | Medium / 中 | High / 高 | Explicit non-goals + RESPONSIBLE_AI refuse-list / 明确非目标 + 拒绝清单 |
| Phase 2 video pipeline cost on a 4090 / Phase 2 视频管线在 4090 上的成本 | Medium / 中 | Medium / 中 | Frozen encoders + short video sample window + Qwen 7B / 冻结编码器 + 小采样窗口 + Qwen 7B |
| Multi-speaker classroom recordings / 多说话人课堂录制 | Medium / 中 | Medium / 中 | Explicit out-of-scope v1; v2 diarization / 明确 v1 不做，v2 引入分离 |

---

## 15. Roadmap / 路线图

| Version / 版本 | Scope / 范围 | Status / 状态 |
| --- | --- | --- |
| **V0** Reference MVP / 参考实现 | Sample interviews · text + visual + HF expression · Streamlit · basic tests / 样例 + 文本+视觉+HF表情 + Streamlit + 基础测试 | ✅ Shipped (Phase 1) / 已上线 |
| **V0.2** Research workflow / 研究流程 | Human review state · Markdown export · filters · bad-case gallery · privacy reminders | Designed / 已设计 |
| **V0.3** Multi-interview synthesis / 多访谈综合 | Aggregation · frequency × severity · opportunity prioritization · contradictions | Planned / 计划中 |
| **V0.4** Evaluation harness / 评估骨架 | Evidence-grounding rubric · actionability rubric · consistency checks · regression tests | Planned / 计划中 |
| **V0.5** Privacy & governance / 隐私与治理 | Redaction · consent status · blocked-use warnings · audit log export | Planned / 计划中 |
| **Phase 2** Multimodal real models / 多模态真模型 | MediaPipe + ViViT + WavLM + Whisper + Qwen · DAiSEE eval · FastAPI + agent schema | 🛠 GPU-ready / 就绪 |

完整 roadmap：`docs/roadmap.md`。Full: `docs/roadmap.md`.

---

## 16. Open questions / 开放问题

| # | Question / 问题 | Owner / 负责人 | Target resolution / 解决时点 |
| --- | --- | --- | --- |
| Q1 | Should the canonical 4-class (DAiSEE) be the user-visible label, or keep the 5 open-vocab labels for nuance? / 用户可见层应该是 4 类还是 5 开放词汇 | PM | Phase 2 user testing / Phase 2 用户测试 |
| Q2 | Sustained-state window — 3 segments? 30 seconds? / 持续状态窗口选 3 段还是 30 秒 | PM | Phase 2 prototype / Phase 2 原型 |
| Q3 | Learned fusion head or heuristic late-fusion for v1? / v1 用学习融合头还是启发式后融合 | Eng | Phase 2 ablation / Phase 2 消融 |
| Q4 | Where does the privacy-sensitive cue threshold live — code, config, or per-customer? / 隐私 cue 阈值放在哪一层 | PM + Eng | Phase 2 |

---

## 17. Appendix — glossary / 附录 — 词汇表

| Term / 术语 | Meaning / 含义 |
| --- | --- |
| Open-vocab emotion label / 开放词汇情绪标签 | One of `frustrated / confused / hesitant / relieved / overwhelmed` / 五选一，比 2 类情感更细 |
| Canonical state (DAiSEE) / 规范状态 | One of `engagement / confusion / frustration / boredom` |
| Evidence / 证据 | An atom with `source`, `quote`, `cue`, `confidence` linking a signal to a perceptible feature / 把信号链接到可感知特征的 atom |
| Structured rationale / 结构化 rationale | A 4-field record (`recognition`, `evidence_attribution`, `product_suggestion`, `safety_note`) / 四字段记录 |
| Signal card / 信号卡 | PM-facing output unit aimed at one of three personas / 面向三类角色之一的输出单元 |
| Safety note / 安全声明 | Required string disclaiming psychological diagnosis; test-pinned / 必填诊断免责，由测试钉住 |
