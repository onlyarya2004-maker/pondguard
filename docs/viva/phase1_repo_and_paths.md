# Viva notes — Phase 1: repository foundation

Owner: M1. Files: `pyproject.toml`, `requirements.txt`, `src/pondguard/paths.py`, `src/pondguard/cli.py`, `.gitignore`, `.github/workflows/ci.yml`, `tests/test_smoke.py`.

## What it does

- **Layout.** A standard `src/` layout. `pip install -e .` installs the `pondguard` package in editable mode and creates the `pondguard` command, which points to `pondguard.cli:main`.
- **Local-only paths (`paths.py`).** When the package is imported, `configure_local_env()`:
  - loads `.env`
  - points `HF_HOME` (the embedding model cache) and `PHOENIX_WORKING_DIR` (the trace store) at `AI team project/cache/`, which is outside the git repo
  - turns off telemetry in ChromaDB, Hugging Face and Phoenix

  It only fills variables that are not already set, so the shell or `.env` always wins. Relative paths are resolved against the repo root.
- **What git never sees (`.gitignore`).**
  - `.env` and `.venv/`
  - SQLite files and the Chroma store
  - model weights (`*.gguf`), which live in `../models` outside the repo anyway
  - the copyrighted KB PDFs (`data/kb/raw/*.pdf`); `data/kb/sources.csv` keeps their links
- **Reproducibility.**
  - `requirements.txt` pins exact versions, with CPU-only torch.
  - `.gitattributes` keeps LF line endings, so Windows and Linux CI see the same files.
  - CI (GitHub Actions) installs the pinned set, runs `ruff check` and runs `pytest` in mock mode on every push.

## Agentic concept shown

None directly. This phase supports the "production thinking" criteria in the guidelines: reproducibility, repo hygiene and code quality (R3 rubric, 2 marks). It also enforces the project's **local-only** rule from the first import.

## Likely viva questions

1. **Why are the models and caches outside the repo?**
   They are several GB each, and a public repo must never contain them. Keeping them in `../models` and `../cache` means a stray `git add -A` cannot commit them. `.gitignore` also blocks `*.gguf` as a second guard.
2. **How do you make sure the embedding model doesn't download into the user's home folder?**
   `pondguard/__init__.py` calls `configure_local_env()` before anything imports sentence-transformers, so `HF_HOME` already points at `AI team project/cache/huggingface`. `tests/test_smoke.py` checks that the caches sit inside the project folder but outside the repo.
3. **Why do the tests run in mock mode?**
   CI has no GPU and no LM Studio, and tests must be deterministic and offline. Mock mode returns fixed, schema-valid responses, so the logic can be tested without a model. Real-model behaviour is checked separately in the smoke test and evaluation runs.
