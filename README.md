# PondGuard

**An agentic early-warning advisor for shrimp-pond disease.** For pond *P* on day *D*, PondGuard combines pond telemetry, the farmer's daily log and published disease manuals into one decision:
- a risk level: Normal, Watch, Alert or Critical
- the likely cause, with citations to the manuals
- a graded action, L0–L4

L3 and L4 actions need human approval (the approval gate is built for Review 4). PondGuard is **advisory only**: confirm any diagnosis with a qualified aquaculture professional or a lab test.

Agentic AI mini-project (26CD7PEAAI), B.E. CSE (Data Science), BMSCE. Team: TODO-TEAM.

> Status: **Review 3 build.** Working: simulator, knowledge base, 8 tools, five agents, the graph, CLI, tracing. Not yet built (Review 4): human approval gate, pond memory, full 40-scenario evaluation, A/B experiment, Streamlit UI.

## How a run works

```
START -> Supervisor --routes--> Monitor        reads 72 h of telemetry, flags anomalies (4 tools)
              ^                 Diagnostician  corrective RAG: search -> grade -> re-query (max 2) -> rank or abstain
              |                 Advisor        risk level + one L0-L4 action, steps cited to the manuals
              |                 Critic         checks citations, banned substances, dosages, action level
              +-- each returns  Finish         builds the report (sources, disclaimer, follow-up question)
```

- The **Supervisor** plans the run, skips diagnosis for a healthy pond, sends a rejected draft back **once**, and stops at 8 steps.
- Every claim cites a knowledge-base passage, or the agent **abstains** and asks one follow-up question.
- Every step is shown live in the terminal and recorded in a trace (JSONL file and the local Phoenix UI).

| Agentic concept | Where it is | Proof |
|---|---|---|
| Tool use | `src/pondguard/tools/` (8 tools, all `@traced_tool`) | `tests/test_tools.py`; tool spans in Phoenix |
| Agentic RAG | `agents/diagnostician.py` (grade, re-query, abstain) | `tests/test_diagnostician.py`; run AMBIG-01 |
| Planning | `agents/supervisor.py` (plan, skip, route, bounds) | run HEALTHY-01; `tests/test_graph.py` |
| Reflection | `agents/critic.py` + `tools/guardrail_tools.py` | run CRITIC-DEMO-01 with fault injection |
| Multi-agent | Supervisor + Monitor, Diagnostician, Advisor, Critic (`graph.py`) | `pondguard run` event log |
| Observability | `tracing.py` (JSONL always, Phoenix optional) | `results/runs/*.jsonl`; Phoenix project `pondguard` |
| Guardrails, bounded loops | `config/settings.yaml`, `config/action_ladder.yaml`, token cap in `llm.py` | `tests/test_graph.py` bound tests |

Details per module are in `docs/viva/`; the concept-to-evidence table is `docs/concept_evidence.md`.

## Runs fully locally

No cloud APIs or online services are used at runtime. The internet is needed once, to download Python packages, models and PDFs.

| Part | Runs on |
|---|---|
| Language models | Small models (≤ 8B, GGUF Q4_K_M) in [LM Studio](https://lmstudio.ai): agent `meta-llama-3.1-8b-instruct`, judge `qwen2.5-7b-instruct` (judge used from Review 4) |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers, CPU |
| Vector store | ChromaDB (persistent, `data/chroma/`) |
| Tracing | Arize Phoenix on `localhost:6006`, plus JSONL files in `results/runs/` |

## Folder layout

Everything lives in the `AI team project` folder. Multi-GB files sit **outside** this git repo, so they can never be committed.

```
AI team project/
├── pondguard/            <- this git repo
│   ├── .venv/            <- Python virtual environment (gitignored)
│   ├── config/  src/  scripts/  tests/  docs/
│   ├── data/             <- scenarios (committed); Chroma store, telemetry (regenerable, gitignored); kb/
│   └── results/          <- smoke-test results (committed); run traces (gitignored)
├── models/               <- LM Studio GGUF model files
└── cache/                <- huggingface (embedding model), phoenix (trace database), pip
```

`data/kb/raw/*.pdf` is gitignored because the knowledge-base PDFs are copyrighted. Their titles and links are in `data/kb/sources.csv`.

## Setup (Windows, Python 3.11)

Run these from the `pondguard/` folder in PowerShell.

```powershell
python -m venv .venv
$env:PIP_CACHE_DIR = "..\cache\pip"
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e . --no-deps
Copy-Item .env.example .env
```

`.env` keeps `LLM_PROVIDER=mock` by default. PondGuard sets `HF_HOME=../cache/huggingface` and `PHOENIX_WORKING_DIR=../cache/phoenix` as soon as the package is imported, so caches stay in the project folder.

**Knowledge base.** Download the PDFs listed in `data/kb/sources.csv` into `data/kb/raw/` as `<id>.pdf` (the Review 3 demo uses `CIBA-EHP-2022` and `CIBA-DISMGMT-2022`), then:

```powershell
.venv\Scripts\pondguard simulate         # writes data/scenarios/*.json and data/telemetry/*.csv
.venv\Scripts\pondguard ingest           # builds the knowledge base (downloads the BGE model once)
```

**Checks** (all tests run in mock mode, offline, with no model loaded):

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\python -m pytest
```

## Demo: run the agents on the real model

1. **LM Studio** (server and model, no GUI needed):

   ```powershell
   lms server start --port 1234
   lms load meta-llama-3.1-8b-instruct --identifier meta-llama-3.1-8b-instruct --gpu max -c 8192 -y
   lms ps                                   # confirm the model is loaded
   ```

2. **Phoenix** (leave this terminal open; UI at http://localhost:6006):

   ```powershell
   .venv\Scripts\pondguard phoenix
   ```

3. **Settings** in `.env`: `LLM_PROVIDER=lmstudio`, `LMSTUDIO_AGENT_MODEL=meta-llama-3.1-8b-instruct`, `LMSTUDIO_JUDGE_MODEL=qwen2.5-7b-instruct`, `TRACING=phoenix`.

4. **Run the four scenarios** (add `--no-cache` to force fresh model calls):

   ```powershell
   .venv\Scripts\pondguard run --scenario HEALTHY-01                           # plan, then diagnosis skipped, L0
   .venv\Scripts\pondguard run --scenario EHP-03                               # tools, corrective RAG, cited EHP diagnosis, L2
   .venv\Scripts\pondguard run --scenario AMBIG-01                             # re-query, then abstain + follow-up question
   .venv\Scripts\pondguard run --scenario CRITIC-DEMO-01 --inject-fault uncited_step   # critic sends the draft back
   ```

   Each run prints every step live, then the report. Open the `pondguard` project in Phoenix to walk through the spans. The JSONL trace path is printed at the end.

**Offline fallback (mock mode).** With `LLM_PROVIDER=mock` the same graph runs with no model, no network and no GPU, after `pondguard ingest --embedder hash`. The mock gives constant outputs (every passage graded 0.5), so it **shows the structure only**, for example re-queries and abstention. Its results are not real: never present them as results. It exists for tests, CI and as an offline backup.

## Limitations

- Telemetry and farmer logs are simulated; scenarios come from documented disease descriptions, not real outbreaks.
- The knowledge base holds two ICAR-CIBA manuals so far. Water-quality limits in `config/thresholds.yaml` are indicative and still marked `TODO-TEAM` (not yet verified against a manual).
- The 8B model's management steps are cited but can be generic or questionable; they are advisory text, not verified practice.
- Tuning was done on the demo scenarios (EHP-03, AMBIG-01, CRITIC-DEMO-01); see `docs/tuning_log.md`.

## Citations

To be completed: datasets, knowledge-base documents (`data/kb/sources.csv`), models and libraries with versions (`requirements.txt`).

## Contributions

TODO-TEAM: member names and who owns which module.
