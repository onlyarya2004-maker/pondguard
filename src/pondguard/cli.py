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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
