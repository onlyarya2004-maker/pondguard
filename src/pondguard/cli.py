"""Command-line entry point: `pondguard <command>`.

Subcommands are added phase by phase (simulate, ingest, warmup, run, approve,
pending). So far:
- `phoenix`     start the local Phoenix tracing UI with its data in ../cache/phoenix
- `trace-demo`  run one dummy traced tool call (checks JSONL and Phoenix tracing)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid

from pondguard import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pondguard",
        description=(
            "PondGuard: agentic early-warning advisor for shrimp-pond disease (advisory only)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"pondguard {__version__}")
    sub = parser.add_subparsers(dest="command", title="commands", metavar="<command>")

    phoenix = sub.add_parser("phoenix", help="start the local Phoenix tracing UI")
    phoenix.add_argument("--port", type=int, default=6006)

    sim = sub.add_parser("simulate", help="generate scenario JSON and telemetry CSV")
    sim.add_argument("--specs", default="config/scenario_specs.yaml")
    sim.add_argument("--out", default="data/scenarios")

    ing = sub.add_parser("ingest", help="build the disease knowledge base from the KB PDFs")
    ing.add_argument(
        "--sources",
        nargs="+",
        default=["CIBA-EHP-2022", "CIBA-DISMGMT-2022"],
        help="document ids from data/kb/sources.csv (default: the two R3 manuals)",
    )
    ing.add_argument("--embedder", choices=["bge", "hash"], default="bge")

    run = sub.add_parser("run", help="run the agents on one scenario, printing each step live")
    run.add_argument("--scenario", required=True, help="e.g. EHP-03")
    run.add_argument(
        "--inject-fault",
        choices=["none", "uncited_step"],
        default=None,
        help="uncited_step: the Advisor's first draft gets one uncited step (critic demo)",
    )
    run.add_argument("--no-cache", action="store_true", help="make every model call fresh")

    demo = sub.add_parser("trace-demo", help="record one dummy tool call to JSONL and Phoenix")
    demo.add_argument(
        "--no-phoenix", action="store_true", help="JSONL only, even if TRACING=phoenix"
    )
    return parser


def cmd_phoenix(port: int) -> int:
    """Run `phoenix serve` with PHOENIX_WORKING_DIR and the offline flags already set.

    Importing pondguard set PHOENIX_WORKING_DIR (../cache/phoenix) and turned
    Phoenix telemetry and external resources off; the child process inherits them.
    """
    env = {**os.environ, "PHOENIX_PORT": str(port)}
    print(f"Phoenix data: {env['PHOENIX_WORKING_DIR']}")
    print(f"Phoenix UI:   http://localhost:{port}")
    cmd = [sys.executable, "-m", "phoenix.server.main", "serve"]
    return subprocess.call(cmd, env=env)


def cmd_trace_demo(no_phoenix: bool) -> int:
    from pondguard.tracing import RunTracer, traced_tool

    @traced_tool
    def dummy_check_thresholds(pond_id: str, do_mg_l: float) -> dict:
        """A stand-in tool: flags dissolved oxygen below 4 mg/L."""
        return {"pond_id": pond_id, "do_mg_l": do_mg_l, "flag": do_mg_l < 4.0}

    run_id = f"trace-demo-{uuid.uuid4().hex[:8]}"
    tags = {"pond_id": "P-07", "scenario_id": "TRACE-DEMO", "variant": "B"}
    with RunTracer(run_id, tags=tags, phoenix=False if no_phoenix else None) as tracer:
        with tracer.span("monitor", kind="node", inputs={"pond_id": "P-07"}) as span:
            result = dummy_check_thresholds("P-07", do_mg_l=3.2)
            span.outputs = result
        tracer.event("monitor", "dummy tool flagged low dissolved oxygen", flag=result["flag"])
    print(f"JSONL:   {tracer.path}")
    print(f"Phoenix: {'sent' if tracer.phoenix_active else 'off (JSONL only)'}")
    return 0


def cmd_simulate(specs: str, out: str) -> int:
    from pathlib import Path

    from pondguard.simulator.scenarios import simulate

    written = simulate(Path(specs), Path(out))
    for path in written:
        print(f"wrote {path}")
    return 0


def cmd_ingest(sources: list[str], embedder: str) -> int:
    from pondguard.kb.embeddings import get_embedder
    from pondguard.kb.ingest import ingest, read_sources
    from pondguard.paths import DATA_DIR

    kb = DATA_DIR / "kb"
    docs = read_sources(kb / "sources.csv", kb / "raw", only=sources)
    counts = ingest(docs, get_embedder(embedder))
    for doc_id, n in counts.items():
        print(f"{doc_id}: {n} chunks")
    print(f"knowledge base 'disease_kb_{embedder}' built: {sum(counts.values())} chunks")
    return 0


def format_report(report: dict) -> str:
    lines = [
        "",
        "=" * 70,
        f"PONDGUARD REPORT  pond {report['pond_id']}  {report['date']}  (advisory only)",
        "=" * 70,
        f"Risk level : {report['risk_level'].upper()}",
        f"Action     : {report['action']['action_level']}"
        + ("   (needs human approval)" if report["needs_human"] else ""),
    ]
    if report["hypotheses"]:
        lines.append("Likely causes:")
        for h in report["hypotheses"]:
            lines.append(
                f"  - {h['condition']} ({h['confidence']:.2f}) cites {', '.join(h['citations'])}"
            )
    else:
        lines.append("Likely causes: none confirmed")
    if report["follow_up_question"]:
        lines.append(f"Follow-up question: {report['follow_up_question']}")
    lines.append("Steps:")
    lines += [f"  {i}. {s}" for i, s in enumerate(report["action"]["steps"], start=1)]
    if report["sources"]:
        lines.append("Sources:")
        lines += [f"  [{s['id']}] {s['title'][:60]}, page {s['page']}" for s in report["sources"]]
    if report["issues"]:
        lines.append("Unresolved review issues: " + "; ".join(report["issues"]))
    lines.append(report["disclaimer"])
    return "\n".join(lines)


def cmd_run(scenario_id: str, inject_fault: str | None, no_cache: bool) -> int:
    from pondguard.config import load_settings
    from pondguard.graph import run_scenario
    from pondguard.llm import get_llm
    from pondguard.paths import DATA_DIR
    from pondguard.simulator.scenarios import load_scenario
    from pondguard.tools.kb_tools import get_retriever

    if inject_fault:
        os.environ["PONDGUARD_INJECT_FAULT"] = inject_fault
    settings = load_settings()
    scenario = load_scenario(scenario_id, DATA_DIR / "scenarios")
    llm = get_llm("agent", settings, use_cache=not no_cache)
    retriever = get_retriever(settings)
    print(f"scenario {scenario_id}  pond {scenario['pond_id']}  model {llm.model} ({llm.provider})")
    if settings.fault_injection != "none":
        print(f"!! fault injection ON: {settings.fault_injection}")
    final, tracer = run_scenario(
        scenario,
        llm,
        settings,
        retriever,
        echo=lambda e: print(f"  [{e['node']}] {e['message']}", flush=True),
    )
    print(format_report(final["report"]))
    print(
        f"\nsteps {final.get('step_count')}  llm calls {llm.budget.calls}  "
        f"tokens {llm.budget.used}  trace {tracer.path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args.scenario, args.inject_fault, args.no_cache)
    if args.command == "ingest":
        return cmd_ingest(args.sources, args.embedder)
    if args.command == "simulate":
        return cmd_simulate(args.specs, args.out)
    if args.command == "phoenix":
        return cmd_phoenix(args.port)
    if args.command == "trace-demo":
        return cmd_trace_demo(args.no_phoenix)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
