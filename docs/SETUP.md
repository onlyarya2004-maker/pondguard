# PondGuard setup guide

For a teammate starting from a fresh `git clone`. Written for **Windows 11 + PowerShell**, the machine the project was built and tested on. Linux and macOS are untested (CI runs the mock-mode tests on Ubuntu only).

Read section 1 first: it tells you what you can do with no downloads at all.

## 1. What needs what

| You want to... | You need |
|---|---|
| Read the code, run `ruff` and the tests (200 pass + 1 skipped without the PDFs; mock mode) | Python 3.11 + the pip packages (sections 3–4). **No models, no GPU, no PDFs, no internet after install.** |
| Generate the scenarios (`pondguard simulate`) | The same. |
| Run the whole agent graph in **mock mode** (structure only) | Python + packages + the two PDFs + `pondguard ingest --embedder hash`. No models, no GPU. |
| Build the real knowledge base (`pondguard ingest`) | The two PDFs + internet **once** (downloads the 128 MB BGE embedding model). CPU only; no LM Studio. |
| Run the agents on the **real model**, live demo, trace walkthrough | Everything: LM Studio, the Llama 3.1 8B model, a GPU (see section 8), the PDFs, Phoenix. |
| Re-run the model smoke test | Everything above plus the Qwen2.5 7B model and the Llama 3.2 3B model. |

Mock mode gives constant outputs (every passage graded 0.5). It shows the structure of a run (plan, re-queries, abstention) but its results are **not real**. Never present them as results.

## 2. What is NOT in the repo, and why

| Not in the repo | Why | Where it comes from |
|---|---|---|
| **Model files** (`.gguf`, ~9.6 GB for two models) | GitHub rejects files over 100 MB; also `*.gguf` is in `.gitignore` | LM Studio download (section 5) |
| **The two CIBA PDFs** (`data/kb/raw/*.pdf`) | Copyrighted; gitignored. The repo keeps only titles and links in `data/kb/sources.csv` | Download (section 6) |
| **The embedding model** (BGE-small, 128 MB) | Downloaded automatically on first use | Hugging Face, cached in `../cache/huggingface` |
| **`.env`** | Local settings, never committed (`.env.example` is the template) | Copy `.env.example` |
| **`*.db` files, `data/chroma/`, `data/telemetry/`** | Regenerable local data | `pondguard simulate` and `pondguard ingest` |
| **Run traces** (`results/runs/`) | Per-run logs, regenerable | Created by every run |
| **`.venv/`** | Machine-specific | Section 3 |

## 3. Python 3.11 and the virtual environment

PondGuard needs **Python 3.11.x** (`pyproject.toml` pins `>=3.11,<3.12`; 3.12 and 3.10 are not supported). Tested with **3.11.9**. Download it from https://www.python.org/downloads/release/python-3119/ ("Windows installer (64-bit)"). Check:

```powershell
py -3.11 --version        # Python 3.11.9
```

Recommended folder layout (the repo's README explains why): put the repo inside a project folder, with `models/` and `cache/` beside it.

```
AI team project/
├── pondguard/     <- git clone goes here
├── models/        <- LM Studio models (optional, see section 5)
└── cache/         <- created automatically (huggingface, phoenix, pip)
```

From inside `pondguard/`:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
```

## 4. Install the pinned packages

```powershell
$env:PIP_CACHE_DIR = "..\cache\pip"     # optional: keeps pip's download cache in the project folder
.venv\Scripts\python -m pip install -r requirements.txt -c constraints-windows.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install --no-deps -r requirements-windows-overrides.txt   # Smart App Control: see docs/SETUP.md
.venv\Scripts\python -m pip install -e . --no-deps
Copy-Item .env.example .env
```

- Takes about 10 minutes and about **2.2 GB** (`.venv`); it downloads CPU-only PyTorch (`torch==2.14.1`).
- `requirements.txt` pins the main packages with `==`. Key pins: `langgraph==1.2.17`, `chromadb==1.5.9`, `sentence-transformers==6.2.0`, `arize-phoenix==20.26.0`, `pydantic==2.14.1`.
- **`constraints-windows.txt` pins every other package** (about 230) to the exact versions of the verified environment. It matters: a fresh install without it picked newer patch releases of about 30 dependencies, and one of them (`bashkit` 0.20.1) was blocked by Smart App Control and broke `pytest`. Always install with `-c constraints-windows.txt` as shown. (CI on Ubuntu does not use it.)
- **`requirements-windows-overrides.txt`** is a second, one-line install step (`--no-deps`) that swaps `arize-phoenix-sqlean` 0.2.1 for 0.2.0 (see the Smart App Control notes just below).
- `.env` keeps `LLM_PROVIDER=mock` until you set up LM Studio (section 5).

### Windows Smart App Control

If Windows **Smart App Control** is on (Settings, Privacy & security, Windows Security, App & browser control), it can block compiled files that are not signed. It did on the team laptop:

- **Smart App Control verdicts change over time.** On 30 Oct 2026 files that had loaded for days started to be blocked: `pandas` 3.1.0 (so `requirements.txt` now pins **`pandas==3.0.2`**), `arize-phoenix-sqlean` 0.2.1 (replaced by **0.2.0** through `requirements-windows-overrides.txt`; `pip check` therefore prints one known mismatch, which is expected), and the `pondguard.exe` launcher.
- **`bashkit`** (needed by Phoenix, and imported when `pytest` loads its plugins): version 0.20.1 was blocked in a fresh install; 0.20.0 loaded on the team laptop. `requirements.txt` and `constraints-windows.txt` now pin **`bashkit==0.20.2`**, whose Windows native module has a timestamped Authenticode signature ([release](https://github.com/everruns/bashkit/releases/tag/v0.20.2)). Confirm this upgrade on the team laptop with Smart App Control still **On**: reinstall using the commands above, then run `.venv\Scripts\python -m pytest --collect-only` followed by `.venv\Scripts\python -m pytest`. Signing and hosted Windows CI do not by themselves establish that machine's policy decision.
- **`ormsgpack`** (needed by LangGraph): version 1.13.0 was blocked. `requirements.txt` pins **`ormsgpack==1.12.2`**, which loads. If you see `ImportError: DLL load failed while importing ormsgpack: An Application Control policy has blocked this file`, make sure that pin is installed.
- **`pytest.exe`** (a launcher program) was blocked at one point. Always run tests as **`python -m pytest`**, which starts Python itself instead of the launcher. `pondguard.exe` is blocked the same way on the team laptop (as of 30 Oct): **use `.venv\Scripts\python -m pondguard.cli <command>` instead of `.venv\Scripts\pondguard <command>` everywhere** (the demo checklist and README already do).
- Do not turn Smart App Control off just for this; it cannot be turned back on without resetting Windows. Use the workarounds above.

## 5. LM Studio and the two models (real-model runs only)

**LM Studio:** download from https://lmstudio.ai. Tested with **LM Studio 0.4.23** and runtime `llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.64.0`. A newer version should work, but every results file records the versions used, and for reproducible results turn **runtime auto-updates off** in LM Studio's settings. The `lms` command-line tool is installed with it, at `%USERPROFILE%\.lmstudio\bin\lms.exe` (add that folder to PATH, or call it by full path).

**The models** (both GGUF, 4-bit `Q4_K_M`, publisher `lmstudio-community`; both ≤ 8B parameters, as the project requires):

| Role | LM Studio model key | File | Size | Source |
|---|---|---|---|---|
| **Agent** | `meta-llama-3.1-8b-instruct` | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | 4.92 GB (4,920,739,168 bytes) | https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF |
| **Judge** (Review 4) | `qwen2.5-7b-instruct` | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 4.68 GB (4,683,073,952 bytes) | https://huggingface.co/lmstudio-community/Qwen2.5-7B-Instruct-GGUF |
| Smoke test only | `llama-3.2-3b-instruct` | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | 2.02 GB | https://huggingface.co/lmstudio-community/Llama-3.2-3B-Instruct-GGUF |

The Review 3 demo needs **only the agent model**. Download with LM Studio's CLI (these exact URLs were checked on 27 Oct 2026 and the sizes above match):

```powershell
lms get "https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" -y
lms get "https://huggingface.co/lmstudio-community/Qwen2.5-7B-Instruct-GGUF/blob/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" -y
lms ls        # must list meta-llama-3.1-8b-instruct and qwen2.5-7b-instruct
```

(Or search the model name in LM Studio's Discover tab and pick the `lmstudio-community` build, `Q4_K_M`.) If `lms ls` shows a **different key**, put that key in `.env` instead.

**Models folder.** By default LM Studio stores models in `%USERPROFILE%\.lmstudio\models`; PondGuard does not care where they are. The team keeps them in `AI team project/models` so everything stays in one folder: in LM Studio open **My Models**, change the models directory, or (with LM Studio closed) edit `downloadsFolder` in `%USERPROFILE%\.lmstudio\settings.json`, then move the `lmstudio-community` folder there and run `lms ls` to confirm it still lists the models.

**Start the server and load the agent model** (no GUI needed):

```powershell
lms server start --port 1234
lms load meta-llama-3.1-8b-instruct --identifier meta-llama-3.1-8b-instruct --gpu max -c 8192 -y
lms ps                                   # shows the loaded model
```

Then edit `.env`:

```
LLM_PROVIDER=lmstudio
LMSTUDIO_AGENT_MODEL=meta-llama-3.1-8b-instruct
LMSTUDIO_JUDGE_MODEL=qwen2.5-7b-instruct
TRACING=phoenix
```

## 6. The knowledge-base PDFs (copyrighted, download yourself)

Save each file in `pondguard/data/kb/raw/` with **exactly** these file names (they must match the `id` column of `data/kb/sources.csv`):

| File name | Title (ICAR-CIBA) | URL | Size | Pages |
|---|---|---|---|---|
| `CIBA-EHP-2022.pdf` | Recent Advances on Diagnosis and Management of Enterocytozoon hepatopenaei (EHP) in Brackishwater Shrimp Aquaculture (TM Series 2021 No. 23, Revised Edition 2022) | https://ciba.res.in/wp-content/uploads/2023/06/Manual_.pdf | 1,482,208 bytes | 110 |
| `CIBA-DISMGMT-2022.pdf` | Training Manual on Disease Management of Brackishwater Aquaculture Farming (TM Series 2022 No. 27) | https://ciba.res.in/wp-content/uploads/2022/07/Final_manual.pdf | 6,370,678 bytes | 205 |

```powershell
New-Item -ItemType Directory -Force data\kb\raw | Out-Null
Invoke-WebRequest "https://ciba.res.in/wp-content/uploads/2023/06/Manual_.pdf"       -OutFile data\kb\raw\CIBA-EHP-2022.pdf
Invoke-WebRequest "https://ciba.res.in/wp-content/uploads/2022/07/Final_manual.pdf"  -OutFile data\kb\raw\CIBA-DISMGMT-2022.pdf
Get-FileHash data\kb\raw\CIBA-EHP-2022.pdf, data\kb\raw\CIBA-DISMGMT-2022.pdf -Algorithm SHA256
```

Expected SHA-256 (as downloaded 25 Oct 2026; if the publisher replaces a file the hash changes and the chunk counts in section 9 may differ):

- `CIBA-EHP-2022.pdf`: `28787f94e933083748c8fe243ea0fb503d3708f487df90ba017d94279930ec1d`
- `CIBA-DISMGMT-2022.pdf`: `68291be1db4f1146febb9f5d6784a3a8f628cd8abb569ea8ca30d4c7a7392109`

Other documents listed in `sources.csv` (FAO papers, WOAH, CAA) are not needed for Review 3.

## 7. Build the data

```powershell
.venv\Scripts\pondguard simulate     # 4 scenarios -> data\scenarios\*.json, data\telemetry\*.csv
.venv\Scripts\pondguard ingest       # real knowledge base; downloads BGE-small (128 MB) once
```

The embedding model caches in **`../cache/huggingface`** (inside the project folder, set automatically by PondGuard), never in your user folder. Chroma writes to `data/chroma/` (about 15 MB).

## 8. Arize Phoenix (trace viewer) and hardware

**Phoenix** is installed by `requirements.txt` (`arize-phoenix==20.26.0`); there is nothing else to download. Start it in its own terminal and leave it open:

```powershell
.venv\Scripts\pondguard phoenix       # UI at http://localhost:6006
```

It stores its data in `../cache/phoenix` and turns off its telemetry. The first start takes about 30 seconds (database setup). If Phoenix is not running, PondGuard logs one warning and traces to the JSONL file only; nothing breaks.

**Hardware (measured on the team laptop; other machines are untested).**

| | Team laptop | Needed |
|---|---|---|
| GPU | NVIDIA RTX 5070 Ti Laptop, 12 GB | Llama 3.1 8B Q4_K_M with 8K context used **5.6 GB VRAM** and +5 GB RAM in LM Studio. Plan on a GPU with **8 GB or more VRAM** for the agent model alone (an estimate, not tested on other cards). Do not load the agent and judge together unless you have 12 GB or more. |
| RAM | 31 GB | **16 GB** or more is a reasonable minimum (estimate): LM Studio about 5 GB, Python plus the embedding model about 2 GB. |
| Disk | | About **13 GB**: models 9.6 GB (two), `.venv` 2.2 GB, caches and data under 0.3 GB. The agent model alone is 4.9 GB. |
| Speed on the team laptop | | About 1.4 s per model call (median); a full run takes under a minute. |

**No GPU?** The embedding model and everything except the language model already run on CPU. Without a GPU you can (a) use mock mode and the tests, or (b) load the model on the CPU with `lms load ... --gpu off` (untested; expect several times slower, so a run may take minutes). Use mock mode for anything you must demo quickly.

## 9. Verify your setup

Run these in `pondguard/`. Each line shows what correct output looks like. Stop at the first step that differs.

**A. No models needed**

```powershell
py -3.11 --version                              # Python 3.11.x
.venv\Scripts\python -m pip check               # one expected line: arize-phoenix 20.26.0 has requirement arize-phoenix-sqlean>=0.2.1, but you have 0.2.0
.venv\Scripts\ruff check .                      # All checks passed!
.venv\Scripts\python -m pytest                  # 200 passed, 1 skipped in about 15 s (the skipped test needs the PDFs; with them in place: 201 passed)
.venv\Scripts\pondguard --version               # pondguard 0.1.0
.venv\Scripts\pondguard simulate                # four lines: wrote data\scenarios\EHP-03.json, HEALTHY-01.json, AMBIG-01.json, CRITIC-DEMO-01.json
```

**B. After the PDFs are in `data/kb/raw/`**

```powershell
.venv\Scripts\pondguard ingest
# CIBA-EHP-2022: 142 chunks
# CIBA-DISMGMT-2022: 297 chunks
# knowledge base 'disease_kb_bge' built: 439 chunks
```

Without the PDFs the command stops with `FileNotFoundError: ... is missing; download it (see sources.csv)`.

**C. Mock-mode run (no model)**

```powershell
.venv\Scripts\pondguard ingest --embedder hash          # offline knowledge base for mock mode
$env:LLM_PROVIDER = "mock"
.venv\Scripts\pondguard run --scenario HEALTHY-01
# ... Risk level : NORMAL    Action : L0    (steps 4, llm calls 2)
```

**D. Real model (LM Studio loaded, `.env` set as in section 5)**

```powershell
curl http://localhost:1234/v1/models            # lists meta-llama-3.1-8b-instruct
curl -s -o NUL -w "%{http_code}" http://localhost:6006/   # 200 once Phoenix is running
.venv\Scripts\pondguard run --scenario EHP-03
```

Correct output ends with a report like:

```
Risk level : ALERT
Action     : L2
Likely causes:
  - ehp (0.90) cites CIBA-EHP-2022:p17:c0 ...
Steps:
  1. Send a sample for PCR/lab testing; closer observation; plus L1 steps [ladder:L2].
  ...
Advisory only. Confirm with a qualified aquaculture professional or lab test.
```

Model wording and the confidence numbers can differ a little between machines, but you should see **ehp** as the top cause, **ALERT / L2**, every step ending in a `[...]` citation, and `[critic] draft passed every check` in the live log. Then open http://localhost:6006, project **pondguard**: you should see the run's spans (nodes, tools, a retrieval round, LLM calls).

The other three demo runs and what to say are in `docs/r3_demo_checklist.md`.

## 10. If something fails

| Symptom | Fix |
|---|---|
| `DLL load failed ... Application Control policy` | Section 4, Smart App Control |
| `pytest.exe` or `pondguard.exe` will not start | Use `python -m pytest` / `python -m pondguard.cli` |
| A new `DLL load failed ... Application Control` error for some package | Find a version that loads: `pip install --no-deps --target %TEMP%	 <package>==<version>`, then `set PYTHONPATH=%TEMP%	` and `python -c "import <module>"`; try older and newer releases, pin the one that works in `requirements.txt` and `constraints-windows.txt`. Meanwhile run the demo in mock mode as a fallback. |
| `Python 3.12` or `3.10` errors on install | Use 3.11.x (section 3) |
| `knowledge base 'disease_kb_bge' not found; run pondguard ingest first` | Section 7 |
| `connection refused` to `localhost:1234` | `lms server start --port 1234`, then `lms load ...` |
| `set LMSTUDIO_AGENT_MODEL in .env` | Section 5 |
| No spans in Phoenix | Is `pondguard phoenix` running, and is `TRACING=phoenix` in `.env`? JSONL files in `results/runs/` are always written |
