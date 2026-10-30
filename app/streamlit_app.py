"""PondGuard dashboard: results, run a scenario, approve paused runs.

Start it from pondguard/:  .venv\\Scripts\\python -m streamlit run app\\streamlit_app.py
(LM Studio's server and the model must be running to run scenarios on the real model.)

Pages
- Results     the A/B evaluation: metrics, confusion matrices, the decision, the paired comparison
- Run         run any scenario (variant A or B) with the live step list; pauses at the human gate
- Approvals   paused L3/L4 runs: Approve, Edit or Reject, then the run resumes from its checkpoint
- About       purpose, disclaimer, how the agent works
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import pondguard  # noqa: F401  (sets the local-only environment before other imports)
from pondguard.config import load_settings
from pondguard.evaluation.experiment import latest
from pondguard.evaluation.runner import load_golden, read_rows
from pondguard.memory import Memory, open_checkpointer
from pondguard.models import DISCLAIMER
from pondguard.paths import DATA_DIR, RESULTS_DIR

st.set_page_config(page_title="PondGuard", page_icon="🦐", layout="wide")
SETTINGS = load_settings()


def banner() -> None:
    if SETTINGS.runtime.provider == "mock":
        st.warning(
            "MOCK MODE: no language model is used. Outputs are constant placeholders that only "
            "show the structure of a run. They are not results.",
            icon="⚠️",
        )
    st.caption(DISCLAIMER)


# --- Results ----------------------------------------------------------------------------------


def page_results() -> None:
    st.header("Evaluation: single-pass RAG (A) vs agentic RAG (B)")
    try:
        _, sum_a = latest("A")
        _, sum_b = latest("B")
    except FileNotFoundError:
        st.info(
            "No evaluation results yet. Run `python -m pondguard.cli eval --variant A` and `B`."
        )
        return
    sa = json.loads(sum_a.read_text())
    models = sa["models"]["agent"]
    st.write(
        f"Model **{models['model_key']}** ({models['quantization']}), temperature "
        f"{sa['temperature']}, {sa['metrics']['all']['n']} scenarios per variant."
    )
    table = pd.read_csv(RESULTS_DIR / "ab_summary.csv")
    part = st.radio(
        "Scenarios",
        ["all", "untuned"],
        horizontal=True,
        help="'untuned' leaves out the scenarios used while tuning (docs/tuning_log.md)",
    )
    shown = table[table["slice"] == part].drop(columns="slice").set_index("metric")
    st.dataframe(shown, use_container_width=True)
    decision = (RESULTS_DIR / "ab_decision.md").read_text(encoding="utf-8")
    with st.expander("A/B decision and paired comparison", expanded=True):
        st.markdown(decision)
    figures = RESULTS_DIR / "figures"
    cols = st.columns(2)
    for col, name in zip(cols, ("metrics_ab.png", "cost_quality.png"), strict=True):
        if (figures / name).exists():
            col.image(str(figures / name))
    cols = st.columns(2)
    for col, variant in zip(cols, ("A", "B"), strict=True):
        if (figures / f"confusion_{variant}.png").exists():
            col.image(str(figures / f"confusion_{variant}.png"))
    st.subheader("Per scenario")
    variant = st.radio("Variant", ["B", "A"], horizontal=True)
    rows = read_rows(latest(variant)[0])
    columns = ["scenario_id", "group", "label_risk", "pred_risk", "label_action"]
    columns += ["pred_action", "pred_conditions", "abstained", "tokens"]
    frame = pd.DataFrame(rows)[columns]
    frame["pred_conditions"] = frame["pred_conditions"].map(", ".join)
    frame["risk_ok"] = frame["label_risk"] == frame["pred_risk"]
    st.dataframe(frame, use_container_width=True, hide_index=True)


# --- Run --------------------------------------------------------------------------------------


def page_run() -> None:
    from pondguard.graph import run_scenario
    from pondguard.llm import get_llm
    from pondguard.simulator.scenarios import load_scenario
    from pondguard.tools.kb_tools import get_retriever

    st.header("Run a scenario")
    ids = [p.stem for p in sorted((DATA_DIR / "scenarios").glob("*.json"))]
    if not ids:
        st.info("No scenarios. Run `python -m pondguard.cli simulate`.")
        return
    left, mid, right = st.columns(3)
    scenario_id = left.selectbox(
        "Scenario", ids, index=ids.index("EHP-03") if "EHP-03" in ids else 0
    )
    variant = mid.radio("Variant", ["B", "A"], horizontal=True)
    inject = right.checkbox("Inject the uncited-step fault (critic demo)")
    scenario = load_scenario(scenario_id, DATA_DIR / "scenarios")
    log = scenario["farmer_logs"][0]
    st.write(f"Pond **{scenario['pond_id']}**, day {scenario['day_of_culture']}. Farmer reports "
             f"{log['symptoms'] or 'no symptoms'}, feed {log['feed_consumed_pct']}%, "
             f"mortality {log['mortality_count']}.")  # fmt: skip
    if not st.button("Run PondGuard", type="primary"):
        return
    import os

    os.environ["PONDGUARD_INJECT_FAULT"] = "uncited_step" if inject else "none"
    settings = load_settings()
    llm = get_llm("agent", settings, use_cache=False)
    steps = st.container()
    with st.status("Running the agents...", expanded=True) as status:
        final, tracer = run_scenario(
            scenario, llm, settings, get_retriever(settings), variant=variant,
            checkpointer=open_checkpointer(), memory=Memory(),
            echo=lambda e: steps.write(f"**{e['node']}**: {e['message']}"),
        )  # fmt: skip
        status.update(label="Done", state="complete")
    if "__paused__" in final:
        plan = final["action_plan"]
        st.warning(
            f"Paused: action {plan['action_level']} ({plan['risk_level']}) needs human "
            "approval. Open the Approvals page."
        )
        return
    report = final["report"]
    st.subheader(f"Risk {report['risk_level'].upper()}, action {report['action']['action_level']}")
    for h in report["hypotheses"]:
        st.write(
            f"- **{h['condition']}** ({h['confidence']:.2f}), cites {', '.join(h['citations'])}"
        )
    if report["follow_up_question"]:
        st.info(f"Follow-up question: {report['follow_up_question']}")
    st.write("Steps")
    for step in report["action"]["steps"]:
        st.write(f"- {step}")
    st.caption(f"Trace: {tracer.path}")


# --- Approvals --------------------------------------------------------------------------------


def page_approvals() -> None:
    from pondguard.graph import resume_run
    from pondguard.llm import get_llm
    from pondguard.tools.kb_tools import get_retriever

    st.header("Approvals")
    memory = Memory()
    pending = memory.list_pending()
    if not pending:
        st.success("Nothing is waiting for approval.")
        return
    for item in pending:
        thread = item["thread_id"]
        with st.container(border=True):
            st.subheader(
                f"{item['scenario_id']}: action {item['action_level']} ({item['risk_level']})"
            )
            st.caption(f"Thread {thread}, waiting since {item['created_at']}")
            reviewer = st.text_input("Reviewer", key=f"rev-{thread}")
            decision = st.radio("Decision", ["approve", "edit", "reject"], horizontal=True,
                                key=f"dec-{thread}")  # fmt: skip
            answer = {"decision": decision, "reviewer": reviewer or "reviewer"}
            if decision == "edit":
                level = st.selectbox("New action level", ["L2", "L3", "L4"], key=f"lvl-{thread}")
                note = st.text_input("Note for the report", key=f"note-{thread}")
                answer["edited_action"] = {"action_level": level, "note": note}
            if st.button("Submit decision", key=f"go-{thread}"):
                try:
                    final, _ = resume_run(
                        thread, answer, get_llm("agent", SETTINGS), SETTINGS,
                        get_retriever(SETTINGS), checkpointer=open_checkpointer(), memory=memory,
                    )  # fmt: skip
                except ValueError as exc:  # an invalid edit is refused before the run resumes
                    st.error(str(exc))
                    continue
                report = final["report"]
                st.success(f"Resumed. Approval: {report['approval_status']}, "
                           f"action {report['action']['action_level']}.")  # fmt: skip
                for step in report["action"]["steps"]:
                    st.write(f"- {step}")


# --- About ------------------------------------------------------------------------------------


def page_about() -> None:
    st.header("About PondGuard")
    st.write(
        "PondGuard is an agentic early-warning advisor for shrimp-pond disease. For one pond and "
        "day it combines telemetry, the farmer's log and published disease manuals into a risk "
        "level, a cited likely cause and a graded action (L0 to L4)."
    )
    st.markdown(
        "- A **Supervisor** plans the run and routes four specialists: **Monitor** (tools), "
        "**Diagnostician** (corrective RAG), **Advisor** (rules plus cited steps) and **Critic** "
        "(guardrails, one revision).\n"
        "- **L3 and L4 need a human**: the run pauses at the gate and resumes from a SQLite "
        "checkpoint after a decision.\n"
        "- Everything runs locally on small language models; every step is traced.\n"
        "- It is **advisory only**: confirm any diagnosis with a qualified aquaculture "
        "professional or a lab test. It never gives drug doses and blocks banned substances."
    )


PAGES = {"Results": page_results, "Run": page_run, "Approvals": page_approvals, "About": page_about}
st.sidebar.title("PondGuard")
choice = st.sidebar.radio("Page", list(PAGES))
banner()
PAGES[choice]()
_ = (Path, load_golden)  # kept importable for tests and quick edits
