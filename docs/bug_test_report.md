# Bug test and cross-check (30 Oct 2026)

Done before starting Review 4 work. Method: lint, tests, secret scan of the full git history, edge-case probes of the tools, guardrails, graph and command line, a docs-versus-code check, and a live run of all four scenarios on the real model.

## Found and fixed

| # | Problem found | Fix | Locked in by |
|---|---|---|---|
| 1 | `compute_trends`, `detect_anomalies` and `check_thresholds` crashed on empty telemetry (`KeyError`), and `detect_anomalies` crashed on 1 or 2 readings | Empty input raises a clear `TelemetryError`; very short windows return no anomalies | `tests/test_tools.py` (short windows, empty input) |
| 2 | The "steady rise" rule fired on a 5-hour window of noise (a tiny window was extrapolated to a per-day slope) | Spike, 24 h drop and rise rules need at least 24 readings (`MIN_WINDOW_HOURS`) | `test_very_short_windows_do_not_crash` (1 to 23 readings) |
| 3 | A pond with no telemetry crashed the whole run | The Monitor reports "Telemetry unavailable" and the run continues; the Advisor never calls such a pond `normal`: it is at least `watch` | `test_missing_telemetry_does_not_crash_and_is_not_called_healthy` |
| 4 | The dosage check missed doses written in words ("2 grams per kg", "half a gram per litre") | The pattern now accepts spelled-out units and small number words | `test_spelled_out_dosages_are_caught` |
| 5 | A draft with no action, or no action steps, passed the Critic with no issues | Both now fail with "the draft has no action steps" | `test_a_draft_without_an_action_or_steps_does_not_pass` |
| 6 | Command-line failures printed Python tracebacks (unknown scenario, missing model key, LM Studio not running, bad document id) | `error: ...` one-line messages with a fix hint; `PONDGUARD_DEBUG=1` shows the full traceback | `tests/test_cli.py` |
| 7 | `PONDGUARD_VARIANT=A` (not built until Review 4) silently ran Variant B | The run is refused with a clear message | `test_variant_a_is_refused_not_silently_run_as_b` |
| 8 | No test covered the Critic sending a draft back to the **Diagnostician** (only to the Advisor) | Added a test; the flow works (diagnosis re-runs, one revision) | `test_critic_sending_the_draft_to_the_diagnostician_re_runs_diagnosis` |
| 9 | Stale test counts in `docs/SETUP.md` | Updated: 200 passed + 1 skipped without the PDFs, 201 with them | docs |

## Checked and fine

- `ruff` clean; 201 tests pass (mock mode, offline); `pip check` shows only the one known `arize-phoenix-sqlean` mismatch (documented in SETUP).
- No secrets anywhere in the git history; no personal paths or email in tracked files; nothing large tracked.
- The 4 committed scenario files regenerate identically from `config/scenario_specs.yaml`.
- Every CLI command the docs mention exists; every environment variable the code reads is in `.env.example`; 8 tools as documented; all download URLs (CIBA PDFs, model files, LM Studio, Python) return 200.
- Live run on Llama 3.1 8B (cache off): HEALTHY-01 (normal, L0, diagnosis skipped), EHP-03 (ehp 0.90, alert, L2), AMBIG-01 (re-queries then abstains with a question), CRITIC-DEMO-01 with fault injection (rejected, sent back once, then passes).

## Known limits (not bugs, not fixed)

- AMBIG-01 shows 1 or 2 re-queries depending on the run (the model's grades vary slightly even at temperature 0); the abstention and the follow-up question are always the same.
- The 8B model's management steps are cited but sometimes generic.
- The dosage check only flags a dose near an action verb ("add", "apply", ...); a dose phrased without one is not caught.
- Smart App Control can block other compiled files without warning; SETUP has the recipe for finding a version that loads.
- Thresholds are still unverified (`TODO-TEAM` sources) and team names are placeholders.
