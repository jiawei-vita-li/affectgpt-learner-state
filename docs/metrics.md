# Metrics

## North Star Metric

`Weekly Useful Grounded Insights`

Definition: the weekly count of insights that users mark as useful and that pass an evidence-grounding threshold.

This is better than optimizing for "more generated insights" because it combines usefulness and trust.

## Product Metrics

| Metric | Definition | Demo Proxy |
| --- | --- | --- |
| Activation | User completes the first meaningful insight workflow | Sample loaded -> insights generated -> evidence reviewed |
| Time-to-Insight | Time from interview selection to first readable insight | Local demo target: under 5 seconds |
| Evidence Coverage | Share of insights with linked evidence | Pain points with evidence / total pain points |
| Insight Usefulness | Whether users save, rate, or act on insights | Every opportunity includes action and success metric |

## Quality Metrics

| Metric | Definition | Demo Proxy |
| --- | --- | --- |
| Evidence Grounding | Insight claim can be traced to source evidence | Each pain point includes transcript/cue evidence |
| Emotion Consistency | Emotion label matches segment evidence | Emotion label generated from transcript + mock cues |
| Actionability | Insight maps to a concrete next step | Every pain point has recommended next step |

## Evaluation Plan

Future evaluation should include:

- Human review of whether evidence supports each insight
- Researcher rating of insight usefulness
- Bad-case taxonomy for weakly grounded or overconfident insights
- Comparison between generic summary and evidence-grounded summary

## Safety Metrics

- Unsupported diagnostic language rate
- High-stakes decision recommendation rate
- Evidence mismatch rate
- Overconfidence rate
- Privacy-sensitive data exposure rate

