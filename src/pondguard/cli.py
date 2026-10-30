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
from pondguard.memory import Memory, open_checkpointer


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
    run.add_argument(
        "--variant",
        choices=["A", "B"],
        default=None,
        help="A: single-pass RAG (experiment baseline); B: agentic RAG (default)",
    )

    appr = sub.add_parser("approve", help="resume a run that is paused at the human gate")
    appr.add_argument("--thread", required=True, help="thread id shown by `pending`")
    appr.add_argument("--decision", required=True, choices=["approve", "edit", "reject"])
    appr.add_argument("--level", help="edit: the new action level (e.g. L4)")
    appr.add_argument("--note", default="", help="edit: a note for the report")
    appr.add_argument("--reviewer", default=os.environ.get("USERNAME", "reviewer"))

    sub.add_parser("pending", help="list runs waiting for human approval")

    jd = sub.add_parser("judge", help="score citation faithfulness with the judge model")
    jd.add_argument("--variant", required=True, choices=["A", "B"])
    jd.add_argument("--no-swap", action="store_true", help="do not unload/load models in LM Studio")
    sub.add_parser("judge-sheet", help="write the 10-claim sheet the team labels by hand")
    sub.add_parser("judge-validate", help="compare the judge with the team's hand labels")
    sub.add_parser("experiment", help="A/B comparison, decision and figures from the latest runs")

    ev = sub.add_parser("eval", help="run the 40-scenario golden set for one variant")
    ev.add_argument("--variant", required=True, choices=["A", "B"])
    ev.add_argument("--only", nargs="+", help="run only these scenario ids (a quick check)")
    ev.add_argument("--no-cache", action="store_true", help="make every model call fresh")

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


def print_paused(final: dict) -> None:
    plan = final["action_plan"]
    thread = final["__paused__"]["thread_id"]
    print("\n" + "=" * 70)
    print(f"PAUSED: action {plan['action_level']} ({plan['risk_level']}) needs human approval")
    print("=" * 70)
    for i, step in enumerate(plan["steps"], start=1):
        print(f"  {i}. {step}")
    print(f"\nThread: {thread}")
    print("List waiting runs : python -m pondguard.cli pending")
    cmd = 'python -m pondguard.cli approve --thread "' + thread + '" --decision approve'
    print("Approve           : " + cmd)
    print('Or --decision reject (downgrades to L2), or --decision edit --level L4 --note "..."')


def cmd_eval(variant: str, only: list[str] | None, no_cache: bool) -> int:
    from pondguard.evaluation.runner import run_eval

    csv_path, summary_path = run_eval(variant, only=only, use_cache=not no_cache)
    print(f"\nrows    {csv_path}\nsummary {summary_path}")
    return 0


def cmd_judge(variant: str, no_swap: bool) -> int:
    from pondguard.evaluation.experiment import latest
    from pondguard.evaluation.judge import run_judge

    _, summary = latest(variant)
    out = run_judge(variant, summary, swap_models=not no_swap)
    print(f"\nwrote {out}")
    return 0


def cmd_judge_sheet() -> int:
    from pondguard.evaluation.experiment import latest
    from pondguard.evaluation.judge import build_sheet

    _, summary = latest("B")
    import json

    stamp = json.loads(summary.read_text(encoding="utf-8"))["started"]
    sheet = build_sheet(summary.parent / f"judge_B_{stamp}.json")
    print(
        f"label the claims in {sheet} (write yes or no in human_supported), then run judge-validate"
    )
    return 0


def cmd_judge_validate() -> int:
    from pondguard.evaluation.judge import validate_judge

    result = validate_judge()
    print(f"judge agrees with the team on {result['agreed']}/{result['cases']} claims")
    for d in result["disagreements"]:
        print(
            f"  {d['claim_id']}: judge {d['judge_said']}, human {d['human_said']}: "
            f"{d['claim'][:80]}"
        )
    return 0


def cmd_experiment() -> int:
    from pondguard.evaluation.experiment import run_experiment
    from pondguard.evaluation.plots import make_figures

    result = run_experiment()
    d = result["decision"]
    print(
        f"decision: adopt {d['adopt']}  (top-3 gain {d['top3_gain_points']} points, "
        f"cost {d['cost_ratio']}x)"
    )
    for path in make_figures():
        print(f"wrote {path}")
    print("wrote results/ab_summary.csv and results/ab_decision.md")
    return 0


def cmd_pending() -> int:
    rows = Memory().list_pending()
    if not rows:
        print("nothing is waiting for approval")
        return 0
    for r in rows:
        print(
            f"{r['thread_id']}  {r['scenario_id']}  risk {r['risk_level']}  "
            f"action {r['action_level']}  since {r['created_at']}"
        )
    return 0


def cmd_approve(thread: str, decision: str, level: str | None, note: str, reviewer: str) -> int:
    from pondguard.config import load_settings
    from pondguard.graph import resume_run
    from pondguard.llm import get_llm
    from pondguard.tools.kb_tools import get_retriever

    settings = load_settings()
    answer: dict = {"decision": decision, "reviewer": reviewer}
    if decision == "edit":
        answer["edited_action"] = {k: v for k, v in (("action_level", level), ("note", note)) if v}
    llm = get_llm("agent", settings)
    final, tracer = resume_run(
        thread,
        answer,
        llm,
        settings,
        get_retriever(settings),
        checkpointer=open_checkpointer(),
        memory=Memory(),
        echo=lambda e: print(f"  [{e['node']}] {e['message']}", flush=True),
    )
    print(format_report(final["report"]))
    print(f"\ntrace {tracer.path}")
    return 0


def cmd_run(
    scenario_id: str, inject_fault: str | None, no_cache: bool, variant: str | None = None
) -> int:
    from pondguard.config import load_settings
    from pondguard.graph import run_scenario
    from pondguard.llm import get_llm
    from pondguard.paths import DATA_DIR
    from pondguard.simulator.scenarios import load_scenario
    from pondguard.tools.kb_tools import get_retriever

    if inject_fault:
        os.environ["PONDGUARD_INJECT_FAULT"] = inject_fault
    settings = load_settings()
    variant = variant or settings.runtime.variant
    scenario_path = DATA_DIR / "scenarios" / f"{scenario_id}.json"
    if not scenario_path.exists():
        known = ", ".join(sorted(p.stem for p in scenario_path.parent.glob("*.json"))) or "none"
        raise CliError(
            f"unknown scenario '{scenario_id}'. Available: {known}. "
            "Generate them with `python -m pondguard.cli simulate`."
        )
    scenario = load_scenario(scenario_id, DATA_DIR / "scenarios")
    llm = get_llm("agent", settings, use_cache=not no_cache)
    retriever = get_retriever(settings)
    memory, checkpointer = Memory(), open_checkpointer()
    print(
        f"scenario {scenario_id}  pond {scenario['pond_id']}  variant {variant}  "
        f"model {llm.model} ({llm.provider})"
    )
    if settings.fault_injection != "none":
        print(f"!! fault injection ON: {settings.fault_injection}")
    final, tracer = run_scenario(
        scenario,
        llm,
        settings,
        retriever,
        variant=variant,
        checkpointer=checkpointer,
        memory=memory,
        echo=lambda e: print(f"  [{e['node']}] {e['message']}", flush=True),
    )
    if "__paused__" in final:
        print_paused(final)
        return 0
    print(format_report(final["report"]))
    print(
        f"\nsteps {final.get('step_count')}  llm calls {llm.budget.calls}  "
        f"tokens {llm.budget.used}  trace {tracer.path}"
    )
    return 0


class CliError(Exception):
    """A problem the user can fix; shown as one line, without a traceback."""


def explain(exc: BaseException) -> str:
    """One-line message with a hint for the failures a teammate is most likely to hit."""
    from pondguard.llm import LLMConfigError

    text = str(exc) or type(exc).__name__
    names = {c.__name__ for c in type(exc).__mro__}
    if isinstance(exc, CliError):
        return text
    if isinstance(exc, LLMConfigError):
        return f"{text} (see docs/SETUP.md, section 5)"
    if names & {"OpenAIConnectionError", "APIConnectionError", "ConnectError", "ConnectionError"}:
        base = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        return (
            f"cannot reach the language model server at {base}. Start LM Studio's server and "
            "load the model: `lms server start --port 1234` then `lms load <model-key> -y` "
            "(docs/SETUP.md, section 5). To run without a model, set LLM_PROVIDER=mock."
        )
    if isinstance(exc, FileNotFoundError) and "kb" in text.lower() and ".pdf" in text.lower():
        return (
            f"{text}. Download the PDFs listed in data/kb/sources.csv (docs/SETUP.md, section 6)."
        )
    if isinstance(exc, RuntimeError) and "pondguard ingest" in text:
        return f"{text} (docs/SETUP.md, section 7)"
    return f"{type(exc).__name__}: {text}"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(parser, args)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # show a short message; PONDGUARD_DEBUG=1 shows the traceback
        if os.environ.get("PONDGUARD_DEBUG"):
            raise
        print(f"error: {explain(exc)}", file=sys.stderr)
        return 1


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.command == "judge":
        return cmd_judge(args.variant, args.no_swap)
    if args.command == "judge-sheet":
        return cmd_judge_sheet()
    if args.command == "judge-validate":
        return cmd_judge_validate()
    if args.command == "experiment":
        return cmd_experiment()
    if args.command == "eval":
        return cmd_eval(args.variant, args.only, args.no_cache)
    if args.command == "pending":
        return cmd_pending()
    if args.command == "approve":
        return cmd_approve(args.thread, args.decision, args.level, args.note, args.reviewer)
    if args.command == "run":
        return cmd_run(args.scenario, args.inject_fault, args.no_cache, args.variant)
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
