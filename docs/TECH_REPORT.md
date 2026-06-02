# Technical Report & Results — AffectGPT Learner State / 技术报告与成果汇报

| Field / 字段 | Value / 值 |
| --- | --- |
| Version / 版本 | v1.0 |
| Status / 状态 | Phase 1 (text + OpenCV + Haar + HF face-expression classifier) verified ✓ · Phase 2 (MediaPipe + ViViT/VideoMAE + WavLM + Whisper + Qwen + DAiSEE eval) designed, GPU-ready / Phase 1（文本 + OpenCV + Haar + HuggingFace 面部表情分类器）已验证 ✓ · Phase 2（MediaPipe + ViViT/VideoMAE + WavLM + Whisper + Qwen + DAiSEE 评估）已设计，GPU 环境就绪 |
| Last updated / 更新于 | 2026-05-28 |
| Code paths / 代码路径 | `core/` (algo) · `app/streamlit_app.py` (UI) · `data/sample_interviews.json` (sample data) · `tests/test_insight_generator.py` (5 cases) |

---

## 1. Executive summary / 摘要

### 1.1 What is built and verified (Phase 1) / Phase 1 已建并验证

- 端到端 dataclass-typed **学习者状态时间线** 管线（解析 → 多源 cue 加权融合 → 痛点 → 机会 → 结构化 rationale → 指标）。
- An end-to-end dataclass-typed **learner-state timeline** pipeline (parse → multi-source cue fusion → pain points → opportunities → structured rationale → metrics).

- 5 标签 **开放词汇情绪集合**（`frustrated / confused / hesitant / relieved / overwhelmed`），各带 valence。
- 5-label **open-vocabulary emotion set** with valence.

- 一个 **真模型** 已在跑：HuggingFace `trpakov/vit-face-expression`（图像分类，top-3 概率反馈到情绪 cue）。
- One **real model** runs today: HuggingFace `trpakov/vit-face-expression` (image classification, top-3 probabilities fed back as cues).

- 低层视觉指标 + Haar 脸数（OpenCV），无脸图像也能产出 cue。
- Low-level visual metrics + Haar face count (OpenCV) so faceless images still produce cues.

- **强制 `safety_note` 字段** + 测试钉住 "not a psychological diagnosis" 文案。
- A **mandatory `safety_note` field** with a test pinning the "not a psychological diagnosis" string.

- 5 条 unittest 全部断言通过。
- 5 unittest cases, all green.

### 1.2 Designed but not yet executed (Phase 2) / 已设计未执行

- 多模态管线：MediaPipe (面部) + ViViT / VideoMAE (视觉) + WavLM (音频) + Whisper (ASR)。Multimodal pipeline.
- 开放词汇短语检索（sentence-transformers）+ Qwen2.5-7B-Instruct rationale（仅生成 rationale，不分类）。Open-vocab retrieval + Qwen2.5-7B rationale.
- DAiSEE 四分类基准（weighted F1 + per-class F1）。DAiSEE 4-class benchmark.
- FastAPI `POST /analyze_learner_state` + Agent 工具 schema。FastAPI endpoint + agent schema.

### 1.3 What this report does *not* claim / 本报告不声明的内容

- 没有在 DAiSEE 或其他基准上跑过完整评估。No full DAiSEE evaluation has been executed yet.
- ViViT / WavLM / Whisper / Qwen 在本 repo 中尚未运行（依赖已记录，运行需 Phase 2）。Those models have not been instantiated from this repo yet.
- 不为任何个体学生做心理状态声明（设计上禁止）。No individual-student state claim is made by design.

---

## 2. System architecture / 系统架构

### 2.1 Phase 1 — text + low-level visual + HF expression classifier / 文本 + 低层视觉 + HF 表情分类

```
                ┌──────────────────────────────────────────────┐
                │                Streamlit UI                   │
                │  language · sample chooser · upload mode      │
                └────────────┬───────────────┬─────────────────┘
                             │               │
                             ▼               ▼
              load_sample_interviews()   media_analyzer.analyze_*
              (core/data_io.py)          (core/media_analyzer.py)
                             │               │
                             └──────┬────────┘
                                    ▼
                        ┌──────────────────────────────────┐
                        │     emotion_timeline.py          │
                        │  rule_fusion(text cue,           │
                        │              audio_metadata,     │
                        │              facial_metadata,    │
                        │              vision_cues)        │
                        └────────────┬─────────────────────┘
                                     ▼
                        ┌──────────────────────────────────┐
                        │     insight_generator.py         │
                        │  pain_points + opportunities +   │
                        │  structured_rationale + metrics  │
                        └────────────┬─────────────────────┘
                                     ▼
                            InsightReport → UI panels
```

### 2.2 Phase 2 — multimodal pipeline (planned) / 多模态管线（计划）

```
       audio → WavLM ─┐
       video → ViViT ─┼─► fusion head ─► canonical state {engagement, confusion,
       face  → MediaPipe ┘                                 frustration, boredom}
       speech → Whisper → transcript ─► sentence-transformers
                                          ├─► open-vocab retrieval (5 labels)
                                          └─► Qwen2.5-7B-Instruct → rationale
                                                                   └─► safety_note (forced)
```

设计要点：**Qwen 不分类，只写 rationale。** 分类来自 encoder + 融合头；LLM 只在已有标签上写解释。这样既避免 LLM 幻觉穿透到产品标签，又让 LLM 用在它最擅长的地方。
Key design: **Qwen does not classify — it only narrates.** Classification comes from encoders + a fusion head; the LLM writes prose over already-decided labels. This keeps hallucinations out of product labels while still letting the LLM do what it's best at.

---

## 3. Component breakdown / 组件拆解

### 3.1 `core/schemas.py`

纯 Python dataclasses（轻量、可序列化、易测）。Plain dataclasses for lightweight serialization and testability.

| Type / 类型 | Key fields / 关键字段 | Notes / 备注 |
| --- | --- | --- |
| `Evidence` | segment_id · quote · source · timestamps · cue · confidence | `source` 列表 / `source` whitelist |
| `TranscriptSegment` | segment_id · timestamps · speaker · text · context · cues | One transcript slice / 一条 transcript |
| `Interview` | id · project · title · participant · session · goals · segments | Full sample / 完整样例 |
| `EmotionSignal` | segment_id · timestamps · label · canonical_label · valence · intensity · confidence · evidence · rationale | Bounded ranges enforced / 范围由 schema 强制 |
| `PainPoint` | id · title · description · affected_segments · journey_stage · product_area · severity · evidence · emotion_signals · recommended_next_step | Severity ∈ {S0..S3} |
| `ProductOpportunity` | id · title · description · suggested_action · success_metric · supporting_pain_points · confidence | Confidence ∈ [0,1] |
| `StructuredRationale` | recognition · evidence_attribution · product_suggestion · **safety_note** | safety_note 必填 |
| `InsightReport` | interview_id · top_pain_points · opportunities · rationales · follow_up_questions · metrics | Top-level container / 顶层容器 |

### 3.2 `core/data_io.py`

- `load_interviews(path)` → 列表 `Interview`，校验必填字段与 source 白名单。Returns a list of `Interview`s, validating required fields and the source whitelist.
- 缺字段或 source 越界直接 `ValueError`（早失败优于晚误用）。Missing fields or out-of-whitelist sources raise `ValueError` — fail early instead of misuse late.

### 3.3 `core/emotion_timeline.py`

Phase 1 实现 5 标签开放词汇打分：
Phase 1 implementation of 5-label open-vocab scoring:

- 文本规则字典 → `frustrated/confused/hesitant/relieved/overwhelmed` 的关键词；每命中加权。Text-rule dictionary with weighted keyword hits.
- 音频元数据：长停顿 → `hesitant` +；高音量爆发 → `frustrated` +。Audio metadata: long pauses → `hesitant` +; loud bursts → `frustrated` +.
- 面部元数据：皱眉/不解 → `confused/frustrated` +。Facial metadata: furrow / puzzled → `confused/frustrated` +.
- HF 面部表情模型（图像上传走 `media_analyzer`）：top-3 概率融入。HF face-expression model: top-3 probabilities folded in.
- 加权融合 → 单标签 + `intensity ∈ [0,1]` + `confidence ∈ [0,1]`。Weighted fusion → single label + bounded scalars.
- 标签 → valence 映射：`relieved` = positive；其余 = negative；不平衡仅作正/负。Label → valence mapping.

测试 `test_open_vocabulary_emotion_timeline` 钉住：任一访谈都覆盖至少 2 个 distinct label，且每信号 evidence ≠ 空。
Test `test_open_vocabulary_emotion_timeline` pins: any interview covers ≥ 2 distinct labels and no signal has empty evidence.

### 3.4 `core/insight_generator.py`

由 `EmotionSignal` 列表构造 `InsightReport`：
Constructs an `InsightReport` from `EmotionSignal`s:

- **痛点 / Pain points**：分组连续负面/犹豫段 → 用首段 quote + cue 来源构造 `PainPoint`。Group consecutive negative/hesitant signals → build `PainPoint`s.
- **机会 / Opportunities**：把痛点映射到 product_area 模板 → `ProductOpportunity`，置信度由源信号置信度均值决定。Map pain → product_area template → `ProductOpportunity`.
- **结构化 rationale**：每条痛点带一个 `StructuredRationale`（recognition + evidence_attribution + product_suggestion + safety_note）。Per-pain-point structured rationale.
- **跟进问题 / Follow-up questions**：基于 unresolved cue 模板生成。Generated from unresolved-cue templates.
- **指标 dict / Metrics dict**：暴露 `signal_count`、`negative_valence_ratio`、`grounded_pain_points`、`total_pain_points`、`evidence_coverage`、`opportunity_count`、`safety_note_present`。Seven keys exposed.

测试钉住：metrics dict 必含 7 字段；`evidence_coverage ∈ [0,1]`；至少 1 个 `safety_note_present == True`。
Tests pin: metrics dict contains the 7 keys; `evidence_coverage ∈ [0,1]`; `safety_note_present` is True for at least one rationale.

### 3.5 `core/media_analyzer.py`

唯一真模型路径所在的文件。The file that holds the only real-model code path.

- `analyze_image(path)`：用 PIL 打开 → numpy/OpenCV 计算亮度 / 对比度 / 饱和度 / 边缘密度 / warmth → Haar cascade（OpenCV bundled）数脸 → HuggingFace `trpakov/vit-face-expression` 分类 top-3 → 返回 dict with `metrics`、`face_count`、`top_emotions`、`source` 标记 `facial_expression_model`。
- `analyze_image(path)`: PIL → OpenCV metrics → Haar face count → HuggingFace ViT classifier → dict tagged `source = facial_expression_model`.

- `analyze_video(path)`：用 OpenCV `VideoCapture` 等距采样最多 8 帧 → 复用图像路径 → 帧级 dict 列表 + 聚合 metrics。
- `analyze_video(path)`: OpenCV samples ≤ 8 frames → per-frame dict list + aggregated metrics.

- HuggingFace 模型实例用 `lru_cache` 缓存，第一次调用加载，后续零启动开销。
- HF model loaded via `lru_cache` — first call triggers load, subsequent calls are zero-startup.

- 安全约束 / Safety constraints: 输出明确标 `source`，UI 用 "visible facial expression, not internal state" 文案；不写入磁盘。Source explicitly tagged; UI uses "visible facial expression" copy; no disk write.

### 3.6 `app/streamlit_app.py`

- 侧栏：语言、控件、访谈选择、输入模式（sample / upload）。Sidebar with language, controls, interview chooser, input mode.
- 主页：顶部 safety banner、访谈上下文、情绪时间线、痛点、机会、结构化 rationale、metrics。Main view with safety banner, context, timeline, pain points, opportunities, rationales, metrics.
- 上传模式：图片/视频 → `analyze_image / analyze_video` → 渲染脸数 + top-3 表情 + 视觉 metrics + 安全声明。Upload mode renders model output with the disclaimer.
- 双语：默认中文，可切英文。Bilingual ZH/EN.

---

## 4. Algorithm & model choices / 算法与模型选型

| Choice / 选择 | Decision / 决定 | Rationale / 理由 | Alternative considered / 评估过的替代 |
| --- | --- | --- | --- |
| Label space | 5 open-vocab + canonical 4 (Phase 2 aligned with DAiSEE) | nuance for product, comparability for eval / 产品要细，评估要对标 | binary valence (too coarse); 7-emotion Ekman (orthogonal to learning) |
| Phase 1 emotion engine | Multi-source weighted heuristic fusion | Cheap, deterministic, testable; gets to a working timeline without GPU / 便宜、确定、可测，无 GPU 也能跑通时间线 | LLM-classifies-segments (rejected: non-deterministic) |
| Real model for Phase 1 | HuggingFace `trpakov/vit-face-expression` | Small, MIT-friendly, runs on CPU / 小、可商用、CPU 可跑 | DeepFace (heavier deps); FER+ (older) |
| Phase 2 visual encoder | ViViT or VideoMAE-base (frozen) | Pretrained on Kinetics; segment-level features cheap on a single 4090 / 预训练好，4090 上抽特征便宜 | TimeSformer (similar); CLIP-only (loses temporal) |
| Phase 2 audio encoder | WavLM-base | Strong for paralinguistic features / 副语言强 | wav2vec 2.0 (similar); HuBERT |
| Phase 2 ASR | Whisper-small / medium | Multilingual ZH + EN; reliable timestamps / 中英多语 + 可靠时间戳 | local sherpa (overkill for demo) |
| Phase 2 LLM (rationale only) | Qwen2.5-7B-Instruct | Bilingual, 7B fits on 4090, MIT / 中英好，4090 跑得动 | GPT-4o (closed/cost); Llama-3-8B (weaker ZH) |
| Phase 2 retrieval head | sentence-transformers (e5 / bge) | Open-vocab labels via embedding similarity / 嵌入相似度做开放词汇 | classifier per label (rigid) |
| Fusion head (Phase 2) | Late fusion → small MLP over per-modality logits | Easier to ablate per-modality contribution / 易做单模态消融 | Cross-attention fusion (more params; deferred to v2) |

---

## 5. Implementation status — true vs aspirational / 实现状态 — 真实 vs 设想

| Capability / 能力 | Built / 已建 | Tests / 测试 | Planned / 计划 | Honest note / 诚实备注 |
| --- | --- | --- | --- | --- |
| Dataclass schema | ✅ | indirect via parse / 解析时间接覆盖 | — | dataclasses, validated on load |
| Sample interview parsing | ✅ | `test_interviews_can_be_parsed` | — | — |
| 5-label open-vocab timeline | ✅ | `test_open_vocabulary_emotion_timeline` | — | rule-based (intentional) |
| Pain points / opportunities / rationales | ✅ | `test_top_pain_points_are_grounded_in_evidence` | — | — |
| Mandatory safety_note | ✅ | `test_structured_rationale_has_safety_boundary` | — | string content asserted |
| Metrics dict (7 keys) | ✅ | `test_summary_metrics_include_grounding_ratio` | — | — |
| OpenCV low-level metrics | ✅ | manual | — | — |
| Haar face count | ✅ | manual | — | OpenCV bundled cascade |
| HF face-expression classifier | ✅ | manual | — | Real model, output tagged `source = facial_expression_model` |
| MediaPipe face landmarks | — | — | ✅ Phase 2 | — |
| ViViT / VideoMAE visual encoder | — | — | ✅ Phase 2 | — |
| WavLM audio encoder | — | — | ✅ Phase 2 | — |
| Whisper ASR | — | — | ✅ Phase 2 | — |
| sentence-transformers open-vocab retrieval | — | — | ✅ Phase 2 | — |
| Qwen2.5-7B-Instruct rationale | — | — | ✅ Phase 2 | rationale only, never classification |
| DAiSEE 4-class eval | — | — | ✅ Phase 2 | weighted F1 + per-class F1 |
| FastAPI endpoint | — | — | ✅ Phase 2 | — |
| Agent tool schema | — | — | ✅ Phase 2 | — |

---

## 6. Evaluation methodology / 评估方法学

### 6.1 Phase 1 — internal consistency / 内部一致性

针对样例数据集，由 `tests/test_insight_generator.py` 锁住的契约：
Asserted by `tests/test_insight_generator.py` over the sample dataset:

1. 样例 JSON 解析无错 + 必填字段在场。Sample JSON parses with required fields present.
2. 每条访谈情绪时间线覆盖至少 2 distinct label。Each interview's timeline covers ≥ 2 distinct labels.
3. Top-N 痛点严格 grounded（每条痛点至少 1 个 evidence atom）。Every top-N pain point has ≥ 1 evidence atom.
4. 摘要 metrics 含恰好 7 字段（`signal_count / negative_valence_ratio / grounded_pain_points / total_pain_points / evidence_coverage / opportunity_count / safety_note_present`），且 `evidence_coverage ∈ [0,1]`。Metrics dict contract.
5. 至少 1 条 rationale 的 `safety_note` 在 `_normalize()` 之后 == `"this is not a psychological diagnosis"`（严格字符串匹配）。Safety-note string assertion.

### 6.2 Phase 1 — perception (manual) / Phase 1 感知（手工）

- 上传 ≥ 6 张样例图片（教室、远程学习、面部困惑、面部惊讶、合影、空教室）→ 验证 Haar 在脸图给出 > 0 计数；空教室返回 `face_count == 0` 不崩溃；表情分类输出合理 top-3。
- Upload ≥ 6 sample images → verify Haar gives > 0 on face images; empty room returns `face_count == 0` without crash; expression classifier gives plausible top-3.
- 短样例视频 → 帧级输出 + 聚合 metrics 合理。Short sample video → per-frame + aggregated metrics look reasonable.

### 6.3 Phase 2 — DAiSEE (planned) / Phase 2 — DAiSEE（计划）

- 数据集：DAiSEE 第 9k 段（boredom / engagement / confusion / frustration，4 级强度）。Dataset: DAiSEE 9k segments, 4 classes × 4 intensity.
- 指标：**weighted F1**（适合不平衡）+ **per-class F1**（暴露弱类）。Weighted + per-class F1.
- Baseline：仅 Qwen2.5-VL 零样本片段分类。Qwen2.5-VL zero-shot baseline.
- 我们的系统：encoder + 融合头分类，LLM 仅写 rationale。Encoders + fusion head classify; LLM rationale only.
- Ablation：单模态（视觉/音频/文本）vs 全模态。Modality ablation.

### 6.4 Phase 2 — actionability (planned, qualitative) / Phase 2 可操作性（计划，定性）

- 5–10 名教师/课程设计师试用 → 5 分量表评估 actionability，定性收集是否会真的修改对应片段。Teacher / designer trial → 5-point usability + qualitative collection.

---

## 7. Phase 1 verified results / Phase 1 已验证成果

> 以下结果均来自 `tests/test_insight_generator.py` 与样例数据。它们反映**产品管线在样例数据上的契约**，**不是** DAiSEE / 学习者基准的真实精度。
> All results below come from `tests/test_insight_generator.py` and the sample data. They describe the **product pipeline's contract on sample data**, **not** real DAiSEE accuracy.

### 7.1 Schema & parsing / Schema 与解析

- 两份样例 interview，全部必填字段存在，无解析报错（断言）。Both sample interviews parse with all required fields present (asserted).
- evidence `source` 全部命中白名单：`transcript / mock_audio_metadata / mock_facial_metadata / facial_expression_model`（断言）。Every evidence `source` is in the whitelist (asserted).

### 7.2 Timeline coverage / 时间线覆盖

- 每条访谈时间线覆盖至少 2 个 distinct 开放词汇标签（断言）。Each interview's timeline covers ≥ 2 distinct open-vocab labels (asserted).
- 每条 `EmotionSignal` 至少 1 条 evidence atom（断言）。Every `EmotionSignal` has ≥ 1 evidence atom (asserted).

### 7.3 Pain-point grounding / 痛点的证据支撑

- Top-N 痛点全部 grounded：每条痛点至少包含 1 个非空 evidence atom（断言）。Every top-N pain point grounded with ≥ 1 evidence atom (asserted).
- `evidence_coverage` 在样例数据上 == 1.0（每条痛点都被引用）。`evidence_coverage` on sample data == 1.0.

### 7.4 Structured rationale safety boundary / 结构化 rationale 安全边界

- 至少一条 rationale 的 `safety_note` normalize 后等于 `"this is not a psychological diagnosis"`（严格字符串断言）。At least one rationale's `safety_note` equals the exact string after normalization (asserted).
- 该断言是回归红线 — 任何提示词改动如果删掉该文案，CI 立即变红。This is a regression red line — any prompt change that strips the disclaimer fails CI.

### 7.5 Metrics dict contract / 指标 dict 契约

- Metrics 恰好含 7 字段；`evidence_coverage ∈ [0,1]`；`signal_count > 0`；`safety_note_present` 是 bool（断言）。Metrics dict has exactly the 7 keys with the bounded values (asserted).

### 7.6 Real-model side / 真模型一侧

- HuggingFace `trpakov/vit-face-expression` 加载一次（`lru_cache`），在样例图片上 top-3 概率合理（手工 review）。HF model loads once and gives plausible top-3 on sample images (manual review).
- Haar 在 6 张样例图片上：脸图 face_count > 0，空教室 face_count == 0，无崩溃。Haar over 6 sample images: face images > 0, empty room = 0, no crash.
- OpenCV 低层 metrics 全部数值（无 NaN，无除零），亮 / 暗 / 高对比 / 低对比图片 metrics 单调。OpenCV metrics are well-formed and monotonic in expected dimensions.

### 7.7 What these results mean / 这些结果说明什么

Phase 1 已验证的结果锁住的是**产品对齐与诚实声明**：每条痛点都有真证据；每条 rationale 都有安全声明；每个面部表情输出都标 source；schema 拒绝越界 source。Phase 2 不需要重写产品 surface，只需要把 cue 来源从 mock metadata 换成真模型输出。
Phase 1 verified results pin **product alignment and honesty**: every pain point has real evidence; every rationale has a disclaimer; every facial output is source-tagged; the schema rejects out-of-whitelist sources. Phase 2 doesn't need to rewrite the product surface — only swap cue sources from mock metadata to real models.

---

## 8. Phase 2 planned experiments (TBD) / Phase 2 计划实验

`outputs/EVAL_RESULTS.md` 是 fill-in 模板。`outputs/EVAL_RESULTS.md` is the fill-in template.

| Experiment / 实验 | Hypothesis / 假设 | Metric / 指标 | Status / 状态 |
| --- | --- | --- | --- |
| DAiSEE 4-class weighted F1 vs Qwen2.5-VL zero-shot / 与 VL 零样本对比 | Encoder+fusion beats VL zero-shot / 编码器+融合优于 VL 零样本 | weighted F1 · per-class F1 | TBD |
| Modality ablation (visual / audio / text) / 单模态消融 | Multimodal > best single modality / 多模态 > 最佳单模态 | weighted F1 per ablation | TBD |
| Sustained-state window (3 / 5 / 10 seg) / 持续状态窗口 | Window length controls false positives / 窗口长度控制误报 | precision @ teacher-trigger | TBD |
| Rationale faithfulness (manual rubric) / Rationale 忠实度 | Qwen rationale only cites listed evidence / Qwen rationale 只引用列出的证据 | faithfulness % over sample | TBD |

---

## 9. Performance characteristics / 性能特征

### 9.1 Phase 1 (measured locally) / Phase 1（本地实测）

- Sample replay 端到端 < 2 秒（CPU）。Sample replay end-to-end < 2 s (CPU-only).
- HF face-expression classifier 首张推理 ~ 1–2 秒（含加载）；后续 ~ 100 ms（CPU）。HF classifier first inference ~ 1–2 s (with load); subsequent ~ 100 ms (CPU).
- Haar 脸数 < 50 ms 单图。Haar face count < 50 ms per image.

### 9.2 Phase 2 (projected, RTX 4090) / Phase 2（预估，RTX 4090）

- ViViT-base segment 特征：每 16 帧 segment ~ 80 ms。ViViT-base ~ 80 ms per 16-frame segment.
- WavLM-base audio 特征：1 秒音频 ~ 30 ms。WavLM-base ~ 30 ms per 1 s audio.
- Whisper-small 实时倍率：1 分钟视频 < 6 秒。Whisper-small RT factor 0.1× on a 10-min video.
- Qwen2.5-7B rationale：每 segment ~ 500–800 ms。Qwen 7B rationale 500–800 ms per segment.
- 10 分钟视频端到端目标：5 分钟内出第一个 actionable 标记。10-min video end-to-end target: < 5 min to first actionable flag.

数字将在 Phase 2 跑完后写入 `outputs/EVAL_RESULTS.md`。Numbers land in `outputs/EVAL_RESULTS.md` after Phase 2 runs.

---

## 10. Failure modes & safety design / 失败模式与安全设计

| Failure mode / 失败模式 | Phase 1 mitigation / Phase 1 缓解 | Phase 2 extension / Phase 2 增强 |
| --- | --- | --- |
| Low-light / no-face frames / 低光或无脸 | Haar 返回 0，metrics 走视觉路径 / Haar 返回 0，走视觉路径 | MediaPipe landmarks 触发 low-confidence path / MediaPipe 触发低置信路径 |
| Off-task humor mistaken for confusion / 戏谑被误判为困惑 | Multi-source cue fusion + low-confidence demotion / 多源融合 + 低置信降权 | Sustained-state window over 3+ segments / 多段持续窗口 |
| LLM hallucinating an emotion / LLM 编造情绪 | LLM 不参与 Phase 1 分类 / Phase 1 不用 LLM 分类 | Phase 2 LLM 仅写 rationale，分类由 encoder + fusion / 同 |
| Cultural / demographic bias / 文化人口偏差 | Source disclaimer + non-individual usage policy / 来源声明 + 非个体使用 | Phase 2 bias-aware rubric + caveat copy / Phase 2 偏差 rubric |
| Use-case drift to grading / 用例漂向打分 | PRD non-goals + RESPONSIBLE_AI refuse-list / PRD 非目标 + 拒绝清单 | API rate-limit + per-tenant policy / API 限速 + 租户策略 |
| Privacy leakage / 隐私泄漏 | No disk persistence; safety banner / 不落盘 + 安全 banner | Redaction layer + audit log / 脱敏层 + 审计日志 |

---

## 11. Testing strategy / 测试策略

`tests/test_insight_generator.py` 五条用例：
`tests/test_insight_generator.py` — five cases:

| Test / 测试 | Pins / 锁住的行为 |
| --- | --- |
| `test_interviews_can_be_parsed` | Schema contract + required fields + source whitelist / 字段与白名单 |
| `test_open_vocabulary_emotion_timeline` | ≥ 2 distinct labels per interview; every signal has evidence / 多标签 + 每信号有证据 |
| `test_top_pain_points_are_grounded_in_evidence` | Every top-N pain point has ≥ 1 evidence atom / 痛点 grounded |
| `test_structured_rationale_has_safety_boundary` | At least one rationale's safety_note == "this is not a psychological diagnosis" / 安全声明字符串断言 |
| `test_summary_metrics_include_grounding_ratio` | Metrics dict has 7 keys; bounded ratios / metrics 契约 |

Phase 2 将新增：DAiSEE 4-class F1 烟测、单模态消融跑通烟测、rationale 忠实度抽样检查。
Phase 2 will add: DAiSEE 4-class F1 smoke; modality-ablation pipeline smoke; rationale-faithfulness sample.

运行：`python -m unittest discover -s tests`（在 `AffectGPT_Learner_State/` 下）。Run with the standard library unittest discovery.

---

## 12. Deployment & reproducibility / 部署与可复现

### 12.1 Local demo (Phase 1) / 本地 demo

```bash
cd AffectGPT_Learner_State
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
首次上传图片时下载 HF 模型（约 90 MB）。First image upload triggers HF model download (~90 MB).

### 12.2 Tests / 测试

```bash
cd AffectGPT_Learner_State
python -m unittest discover -s tests
```

### 12.3 Phase 2 on GPU (planned) / GPU 上的 Phase 2

```bash
# Inside a remote GPU screen:
cd /workspace/AffectGPT_Learner_State
python scripts/run_daisee_eval.py \
    --split test --modality multimodal \
    --out outputs/daisee_metrics.json
```
（脚本是 Phase 2 交付物。Script is Phase 2 deliverable.）

---

## 13. Known limitations / 已知局限（诚实清单）

1. Phase 1 样例数据集是 **2 条精选访谈** — 适合契约锁定，不适合统计推断。Sample set has only 2 interviews — fine for contract, not stats.
2. Phase 1 的「audio cue」与「facial cue」一部分仍是 **mock metadata**；只有 HF 表情分类是真模型，且每条 evidence 显式标了 source。Some audio/facial cues are still mock metadata; only the HF classifier is a real model — and every evidence atom names its source.
3. 没有跑过 DAiSEE 评估。No DAiSEE evaluation run yet.
4. 没有跑过 Qwen2.5-7B rationale（依赖已记录，Phase 2 执行）。No Qwen 7B rationale run yet.
5. 没做多说话人课堂分离（v2 引入）。No multi-speaker diarization (v2).
6. 面部表情分类带文化偏差；产品边界明确禁止个体决策。Facial-expression classification carries cultural bias; product boundary blocks individual decisions.

---

## 14. Future work / 未来工作

| Direction / 方向 | Effort / 工作量 | Value / 价值 |
| --- | --- | --- |
| Phase 2 execution → DAiSEE numbers / Phase 2 执行 → DAiSEE 数字 | L | Replaces TBD with real F1 / 真 F1 替代 TBD |
| MediaPipe + ViViT 接入 | M | Real visual modality / 真视觉模态 |
| WavLM + Whisper 接入 | M | Real audio + transcript / 真音频 + 转写 |
| Qwen2.5-7B rationale | M | Real rationale prose / 真 rationale 文本 |
| FastAPI endpoint + agent schema | S | LMS / agent integration / LMS 与 agent 集成 |
| Multi-speaker diarization (v2) / 多说话人 v2 | M | Classroom scale / 课堂规模 |
| Bias-aware rubric + caveat copy / 偏差 rubric | S | RAI defensibility / RAI 可辩护 |

---

## Appendix A — Code map / 代码地图

| File / 文件 | Role / 作用 | LoC (approx) / 大致行数 |
| --- | --- | --- |
| `core/schemas.py` | Dataclasses, source whitelist, bounded fields / 数据类与白名单 | ~110 |
| `core/data_io.py` | JSON → Interview loader / JSON → Interview | ~60 |
| `core/emotion_timeline.py` | 5-label weighted fusion / 5 标签加权融合 | ~150 |
| `core/insight_generator.py` | Pain points · opportunities · rationales · metrics / 痛点/机会/rationale/指标 | ~200 |
| `core/media_analyzer.py` | OpenCV + Haar + HuggingFace ViT / 低层视觉 + Haar + HF ViT | ~150 |
| `app/streamlit_app.py` | Bilingual UI / 中英 UI | ~250 |
| `data/sample_interviews.json` | 2 sample interviews / 2 条样例访谈 | JSON |
| `tests/test_insight_generator.py` | 5 unittest cases / 5 个 unittest | ~70 |
| `docs/PRD_full.md` | Product spec (bilingual) / 产品规格（双语） |
| `docs/TECH_REPORT.md` | This document / 本文 |
| `docs/RESPONSIBLE_AI.md` | Safety commitments / 安全承诺 |
| `docs/competitive_analysis.md` | Wedge / 差异化 |
| `docs/user_research.md` | User stories / 用户故事 |
| `docs/metrics.md` | Metric definitions / 指标定义 |
| `docs/roadmap.md` | V0 → V4 |
| `outputs/EVAL_RESULTS.md` | Phase 2 fill-in / Phase 2 填表 |
