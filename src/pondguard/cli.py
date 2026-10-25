"""Command-line entry point: `pondguard <command>`.

Subcommands are added phase by phase (simulate, ingest, warmup, run, approve,
pending). Phase 1 only provides the skeleton and `--version`.
"""

from __future__ import annotations

import argparse

from pondguard import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pondguard",
        description=(
            "PondGuard: agentic early-warning advisor for shrimp-pond disease (advisory only)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"pondguard {__version__}")
    parser.add_subparsers(dest="command", title="commands", metavar="<command>")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
