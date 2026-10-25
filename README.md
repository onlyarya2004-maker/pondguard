# PondGuard

**An agentic early-warning advisor for shrimp-pond disease.** For pond *P* on day *D*, PondGuard combines pond telemetry, the farmer's daily log and published disease manuals into one decision:
- a risk level: Normal, Watch, Alert or Critical
- the likely cause, with citations
- a graded action, L0–L4

L3 and L4 actions pause for human approval. PondGuard is **advisory only**: confirm any diagnosis with a qualified aquaculture professional or a lab test.

Agentic AI mini-project (26CD7PEAAI), B.E. CSE (Data Science), BMSCE. Team: TODO-TEAM.

> Status: **Phase 1, repository skeleton.** The build plan is `../PondGuard_Build_Plan.md`.

## Runs fully locally

No cloud APIs or online services are used at runtime. The internet is needed once, to download Python packages, models and PDFs.

| Part | Runs on |
|---|---|
| Language models | Small models (≤ 8B, GGUF Q4_K_M) in [LM Studio](https://lmstudio.ai) |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers, CPU |
| Vector store | ChromaDB (persistent, `data/chroma/`) |
| Memory and checkpoints | SQLite (`data/pondguard.db`) |
| Tracing | Arize Phoenix on `localhost:6006`, plus JSONL files in `results/runs/` |
| UI | Streamlit |

## Folder layout

Everything lives in the `AI team project` folder. Multi-GB files sit **outside** this git repo, so they can never be committed.

```
AI team project/
├── pondguard/            ← this git repo
│   ├── .venv/            ← Python virtual environment (gitignored)
│   ├── config/  src/  app/  eval/  tests/  docs/
│   ├── data/             ← Chroma store, SQLite, telemetry, scenarios, kb/
│   └── results/          ← eval CSVs, summaries, figures
├── models/               ← LM Studio GGUF model files
└── cache/
    ├── huggingface/      ← embedding model (HF_HOME)
    ├── phoenix/          ← Phoenix trace database (PHOENIX_WORKING_DIR)
    └── pip/              ← pip download cache
```

`data/kb/raw/*.pdf` is gitignored because the knowledge-base PDFs are copyrighted. Their titles and links are in `data/kb/sources.csv`, so anyone can download them again.

## Setup (Windows, Python 3.11)

Run these from the `pondguard/` folder in PowerShell.

```powershell
# 1. Virtual environment inside the repo; pip cache inside the project
python -m venv .venv
$env:PIP_CACHE_DIR = "..\cache\pip"
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e . --no-deps

# 2. Local settings (mock mode by default)
Copy-Item .env.example .env
```

`.env` sets `HF_HOME=../cache/huggingface` and `PHOENIX_WORKING_DIR=../cache/phoenix`. PondGuard applies these as soon as the package is imported, so the embedding model and traces stay in the project folder. Relative paths in `.env` are resolved against `pondguard/`.

## Checks

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\pytest
.venv\Scripts\pondguard --help
```

All tests run in **mock mode**, offline, with no model loaded.

## Phoenix (local tracing UI)

Start Phoenix with PondGuard's wrapper. It stores data in `../cache/phoenix` and switches Phoenix telemetry and external resources off. Leave this terminal open:

```powershell
.venv\Scripts\pondguard phoenix          # UI at http://localhost:6006
```

Then set `TRACING=phoenix` in `.env` and check tracing end to end:

```powershell
.venv\Scripts\pondguard trace-demo       # one dummy tool call -> JSONL + Phoenix project "pondguard"
```

Every run is also written to `results/runs/{run_id}.jsonl`, whether or not Phoenix is running. If Phoenix is down, PondGuard logs one warning and continues with JSONL only.

## LM Studio

LM Studio's models folder is set to `AI team project/models`. Start the server and load a model without the GUI:

```powershell
lms server start --port 1234
lms load <model-key> --identifier <model-key> --gpu max -y
lms ps              # confirm the model is loaded
lms server status
```

Then set `LLM_PROVIDER=lmstudio` and `LMSTUDIO_AGENT_MODEL` / `LMSTUDIO_JUDGE_MODEL` in `.env`. Candidate models (all Q4_K_M):
- `qwen2.5-7b-instruct`
- Llama 3.1 8B Instruct
- Llama 3.2 3B Instruct

The agent and judge models are chosen in the Phase 5 smoke test (`docs/compute_budget.md`).

## Citations

To be completed: datasets, knowledge-base documents (`data/kb/sources.csv`), models and libraries with versions.
