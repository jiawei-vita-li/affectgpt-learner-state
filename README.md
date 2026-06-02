# AffectGPT — Learner State Feedback for E-Learning Video

> Instructional video / camera feed / sample interview → multimodal **learner-state timeline** with evidence + three downstream signal cards (course design / live-class teacher / adaptive engine).
>
> 教学视频 / 摄像头流 / 样例访谈 → 带证据的**学习者状态时间线** + 三类下游信号卡（课程设计 / 直播教师 / 自适应引擎）。

**Hard boundary**: this is a learning-experience product. It is **not for** student grading, hiring or admissions filtering, exam scoring, or clinical psychological diagnosis.
**硬边界**：本项目是学习体验产品，**不用于**学生打分、招聘/招生筛选、考试评分或临床心理诊断。

---

## What you get / 仓库里有什么

| Layer / 层 | Content / 内容 |
| --- | --- |
| Algorithm / 算法 | 5-label open-vocabulary emotion set · multi-source weighted cue fusion · HuggingFace `trpakov/vit-face-expression` for facial-expression classification (top-3) · OpenCV low-level visual metrics + Haar face count |
| Product / 产品 | Bilingual Streamlit (ZH/EN) · sample-interview replay + image/video upload · pain points · product opportunities · structured rationale with mandatory `safety_note` · grounding metrics surfaced as a dict |
| Engineering / 工程 | 5 unittest cases pinning the schema contract, evidence grounding, and safety-note string · dataclass-typed schemas · explicit `source` whitelist per evidence atom |

---

## Quick start / 快速开始

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The first image upload triggers a ~90 MB HuggingFace model download.
首次上传图片会触发 ~90 MB 的 HuggingFace 模型下载。

### Run tests / 跑测试

```bash
python -m unittest discover -s tests
```

5 tests pin: schema parsing + source whitelist, ≥ 2 distinct emotion labels per interview with evidence on every signal, top-N pain points all grounded, mandatory `safety_note` string equals `"this is not a psychological diagnosis"`, and the 7-key metrics dict contract.
5 个测试钉住：schema 解析 + source 白名单、每访谈 ≥ 2 distinct 标签且每信号有证据、Top-N 痛点全 grounded、`safety_note` 必含「这不是心理诊断」、7 字段 metrics dict 契约。

---

## Documentation / 文档

- **[docs/PRD_full.md](docs/PRD_full.md)** — Full bilingual PRD (17 sections) / 完整双语 PRD（17 章）
- **[docs/TECH_REPORT.md](docs/TECH_REPORT.md)** — Technical report + verified results + Phase 2 plan / 技术报告 + 已验证成果 + Phase 2 计划
- [docs/PRD.md](docs/PRD.md) — TL;DR entry / 入口
- [docs/metrics.md](docs/metrics.md) · [docs/competitive_analysis.md](docs/competitive_analysis.md) · [docs/user_research.md](docs/user_research.md) · [docs/roadmap.md](docs/roadmap.md) · [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md)
- [outputs/EVAL_RESULTS.md](outputs/EVAL_RESULTS.md) — DAiSEE evaluation (Phase 2 fill-in template) / DAiSEE 评估（Phase 2 填写模板）

---

## Phase split / 阶段划分

The repo is honest about what is verified today vs what is planned.
仓库对「今天已验证」和「计划中」严格区分。

| Phase | What runs today / 当前能跑 | Notes / 备注 |
| --- | --- | --- |
| **Phase 1** (this repo) | Text + low-level visual + HuggingFace face-expression classifier · 5 open-vocab labels mapped to canonical 4-class (DAiSEE-aligned) · structured rationale with safety_note · 5 unittest cases | Every evidence atom names its `source` (`transcript / mock_audio_metadata / mock_facial_metadata / facial_expression_model`) so reviewers cannot mistake placeholder cues for real model output |
| **Phase 2** (designed) | MediaPipe face landmarks · ViViT / VideoMAE visual encoder · WavLM audio · Whisper ASR · sentence-transformers open-vocab retrieval · Qwen2.5-7B for rationale only (never classification) · DAiSEE 4-class weighted F1 + per-class F1 · FastAPI `POST /analyze_learner_state` + agent tool schema | To be added once Phase 2 runs / Phase 2 执行后补 |

Phase 2 numbers not yet executed are labeled `TBD` in `outputs/EVAL_RESULTS.md`.
所有 Phase 2 尚未执行的指标在 `outputs/EVAL_RESULTS.md` 中显式标 `TBD`。

---

## Responsible AI / 负责任 AI

See [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md). PRD-level commitments:
完整内容见 [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md)。要点：

- **Not for** punitive analytics, grading, hiring, exam scoring, or clinical diagnosis. / 不用于惩罚性分析、打分、招聘、考试评分、临床诊断。
- **Mandatory `safety_note`** — `StructuredRationale.safety_note` is a required dataclass field; a unit test pins the string `"this is not a psychological diagnosis"`. / 安全字段强制，单元测试钉住免责字符串。
- **Low-confidence routes to human review**, never auto-notifies the student or teacher. / 低置信路由人工 review，绝不自动通知学生或教师。
- **No raw-video retention** in the demo path. / demo 路径不留存原视频。
- LLM (planned Qwen2.5-7B) is used only to write rationale prose; classification stays with encoders + fusion head — so hallucinations cannot poison product labels. / LLM 仅写 rationale，分类交给编码器与融合头，幻觉无法污染产品标签。

---

## License

MIT — see [LICENSE](LICENSE).
