"""Early check: run the real telemetry tools + Diagnostician on a scenario, live model.

Prints every retrieval round: the query, each passage's score and grade, and the
decision (rank / re-query / abstain), then the final diagnosis. Nothing is scripted:
the grades come from the model named in .env / the environment.

Usage (LM Studio server running, KB ingested, scenarios simulated):
    set LLM_PROVIDER=lmstudio, LMSTUDIO_AGENT_MODEL=..., LMSTUDIO_JUDGE_MODEL=...
    .venv\\Scripts\\python scripts\\diagnose_check.py EHP-03 [--no-cache] [--repeat N]
"""

from __future__ import annotations

import argparse
import json

import pondguard  # noqa: F401
from pondguard.agents.diagnostician import diagnose
from pondguard.config import load_settings
from pondguard.llm import get_llm
from pondguard.paths import DATA_DIR
from pondguard.simulator.scenarios import load_scenario
from pondguard.tools.telemetry_tools import check_thresholds, detect_anomalies, get_telemetry
from pondguard.tracing import RunTracer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--repeat", type=int, default=0, help="cache slot / repeat index")
    args = parser.parse_args()

    settings = load_settings()
    scenario = load_scenario(args.scenario, DATA_DIR / "scenarios")
    log = scenario["farmer_logs"][0]
    llm = get_llm("agent", settings, use_cache=not args.no_cache, repeat=args.repeat)
    print(f"scenario {args.scenario}  pond {scenario['pond_id']}  model {llm.model}")
    print(f"symptoms: {log['symptoms']}  feed {log['feed_consumed_pct']}%")

    run_id = f"diagcheck-{args.scenario}-{args.repeat}"
    with RunTracer(run_id, tags={"scenario_id": args.scenario}) as tracer:
        readings = get_telemetry(scenario["pond_id"], scenario_id=args.scenario)
        anomalies = detect_anomalies(readings) + check_thresholds(readings)
        for a in anomalies:
            print(f"  anomaly: {a.parameter} {a.kind} {a.severity} value={a.value} ({a.window})")
        result = diagnose(
            anomalies=anomalies,
            symptoms=log["symptoms"],
            day_of_culture=scenario["day_of_culture"],
            llm=llm,
            settings=settings,
        )
    for r in result.rounds:
        print(f"\nround {r['round']}: query = {r['query']}")
        for p in r["passages"]:
            print(f"   score {p['score']:.3f}  grade {p['grade']}  {p['id']:28} {p['disease']}")
        total = len(r["passages"])
        print(f"   {r['relevant']}/{total} relevant (need {r['needed']}) -> {r['decision']}")
    d = result.diagnosis
    print("\nDIAGNOSIS:", "abstained" if d.abstained else "")
    for h in d.hypotheses:
        print(f"   {h.condition} {h.confidence:.2f} cites {h.citations}")
    if d.abstained:
        print("   follow-up:", d.follow_up_question)
    print(f"model calls: {len(llm.calls)}  tokens: {llm.budget.used}  trace: {tracer.path}")
    print(json.dumps({"label": scenario["label"]}, indent=None))


if __name__ == "__main__":
    main()
