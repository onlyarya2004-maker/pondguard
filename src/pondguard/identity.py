"""Exact model identity for every results file (plan rule 8).

Agentic concept: reproducible evaluation. A score only means something if we
know exactly which model produced it, so each run records, for the agent and the
judge: the LM Studio model key, GGUF file, quantization, the LM Studio app
version and the selected llama.cpp runtime. All of it comes from the local `lms`
CLI; nothing is guessed. Unknown values are recorded as None, never invented.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelIdentity:
    provider: str
    model_key: str
    file: str | None = None
    quantization: str | None = None
    params: str | None = None
    lmstudio_version: str | None = None
    runtime: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_lms() -> str | None:
    """The `lms` CLI: on PATH, or in its default install folder."""
    found = shutil.which("lms")
    if found:
        return found
    exe = "lms.exe" if sys.platform == "win32" else "lms"
    default = Path.home() / ".lmstudio" / "bin" / exe
    return str(default) if default.exists() else None


def _run(args: list[str], timeout: int = 60) -> str:
    try:
        out = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",  # lms prints UTF-8 (the check mark), not the Windows codepage
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout or ""


def _json_from(text: str) -> Any:
    """`lms` may print a status line before the JSON; take the JSON part."""
    start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=-1)
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return None


def parse_selected_runtime(table: str) -> str | None:
    """Pick the row marked with a check in `lms runtime ls` output."""
    for line in table.splitlines():
        if "✓" in line:
            return line.split()[0]
    return None


@lru_cache(maxsize=1)
def lmstudio_app_version() -> str | None:
    """Product version of the installed LM Studio app (Windows only)."""
    if sys.platform != "win32":
        return None
    exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "LM Studio" / "LM Studio.exe"
    if not exe.exists():
        return None
    out = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(Get-Item '{exe}').VersionInfo.ProductVersion",
        ]
    ).strip()
    return out or None


def lmstudio_identity(model_key: str) -> ModelIdentity:
    lms = find_lms()
    if lms is None:
        return ModelIdentity(provider="lmstudio", model_key=model_key)
    models = _json_from(_run([lms, "ls", "--json"])) or []
    entry = next((m for m in models if m.get("modelKey") == model_key), {})
    quant = (entry.get("quantization") or {}).get("name")
    return ModelIdentity(
        provider="lmstudio",
        model_key=model_key,
        file=entry.get("path"),
        quantization=quant,
        params=entry.get("paramsString"),
        lmstudio_version=lmstudio_app_version(),
        runtime=parse_selected_runtime(_run([lms, "runtime", "ls"])),
    )


def model_identity(provider: str, model_key: str) -> ModelIdentity:
    if provider == "lmstudio":
        return lmstudio_identity(model_key)
    return ModelIdentity(provider=provider, model_key=model_key)
