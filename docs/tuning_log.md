# Tuning log

Every scenario, setting or prompt used to tune the system is recorded here, so the
"untuned" evaluation slice (R4) can be computed honestly. Scenarios listed under
"tuned on" leave the untuned subset.

## Tools (tuned on simulated healthy ponds, not on golden scenarios)

| Date | Change | Evidence |
|---|---|---|
| 27 Oct | Spike rule cutoff set to |z| > 4 instead of the spec's 3 (`SPIKE_Z` in `tools/telemetry_tools.py`) | 300 generated healthy 72 h windows: z > 3 false alarm 37.7%, z > 3.5 5.7%, z > 4 0.3%, z > 4.5 0% |
| 27 Oct | Dawn-oxygen trend counts only if the three minima fall by at least 0.3 mg/L (`MIN_DAWN_DROP`) | A healthy seed fell 0.04 mg/L across three dawns (sensor noise) |
| 27 Oct | Dosage check ignores the word "dissolved" in "dissolved oxygen" | "keep dissolved oxygen above 5 mg/L" was wrongly flagged as a dosage |

## Prompts and agents

(None yet. Phase 6 prompt tuning on the real model is recorded here, with the scenarios used.)

Tuned on: *(none so far)*
