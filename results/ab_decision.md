# A/B decision: single-pass RAG (A) vs agentic RAG (B)

Model: meta-llama-3.1-8b-instruct (Q4_K_M), runtime llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.67.0. Run A `20261030_223911` (commit 1db5a90), run B `20261030_223213` (commit 1db5a90).

## Decision rule
Adopt B if its top-3 hit rate or citation faithfulness rises by at least 5 points and its cost (mean tokens per run) is at most 2x A's. Otherwise keep A.

## Result on all 40 scenarios
- Top-3 hit: A 1.0, B 0.8667 (-13.3 points)
- Faithfulness gain: -10.6 points
- Cost ratio (B tokens / A tokens): 1.27x (A 4579.6, B 5835.125)
- Quality rule met: False. Cost rule met: True.
- **Decision: adopt A.**

## Same rule on the untuned scenarios only
- Top-3 gain -13.8 points, cost ratio 1.27x, decision: adopt A.

## Paired comparison (same scenarios under both variants)
- risk_correct: both right 37, both wrong 0, B fixed A on 3 (AMBIG-01, AMBIG-03, AMBIG-05), B broke A on 0 (none)
- top3_hit: both right 26, both wrong 0, B fixed A on 0 (none), B broke A on 4 (AHPND-03, WQ-01, WQ-02, WSSV-05)
- abstention_correct: both right 33, both wrong 0, B fixed A on 5 (AMBIG-01, AMBIG-02, AMBIG-03, AMBIG-04, AMBIG-05), B broke A on 2 (WQ-01, WQ-02)

## Caveats
- Only 40 scenarios (about 30 with a labelled condition): one scenario is 2.5 to 3.3 points, so a 5-point difference is one or two scenarios. Read the paired lists, not just the averages.
- The scenarios are simulated, and prompts and rules were tuned on EHP-03, AMBIG-01 and HEALTHY-01 (see docs/tuning_log.md), which is why the untuned slice is reported too.
- One run per variant at temperature 0; run-to-run variation was not measured.
