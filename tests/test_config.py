"""Config files load, validate, and enforce the safety invariants."""

import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pondguard.config import RuntimeEnv, load_settings
from pondguard.paths import CONFIG_DIR, REPO_ROOT


@pytest.fixture
def settings():
    return load_settings(env={})


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    for f in CONFIG_DIR.glob("*.yaml"):
        shutil.copy(f, tmp_path / f.name)
    return tmp_path


def _edit(path: Path, change) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_loop_bounds_match_the_plan(settings):
    b = settings.loop_bounds
    assert (b.max_supervisor_steps, b.max_requeries, b.max_critic_revisions) == (8, 2, 1)
    assert b.recursion_limit == 25
    assert b.token_cap_per_run == 20000


def test_retrieval_and_llm_settings(settings):
    assert settings.retrieval.k == 5
    assert settings.retrieval.grade_threshold == 0.6
    assert settings.retrieval.min_relevant_passages == 2
    assert settings.llm.temperature == 0


def test_every_threshold_has_a_source_and_none_is_verified_yet(settings):
    assert set(settings.thresholds) >= {"do_mg_l", "ph", "temp_c", "tan_mg_l", "nitrite_mg_l"}
    for spec in settings.thresholds.values():
        assert spec.source
        assert not spec.verified  # flips once the team fills real citations


def test_action_ladder_rules(settings):
    ladder = settings.action_ladder
    assert ladder.allowed_actions["normal"] == ["L0"]
    assert ladder.allowed_actions["critical"] == ["L2", "L3", "L4"]
    assert ladder.requires_approval("L3") and ladder.requires_approval("L4")
    assert not ladder.requires_approval("L2")
    assert ladder.condition_default_risk["wssv"] == "critical"
    assert ladder.escalation.consecutive_days == 3


def test_l3_without_approval_is_rejected(config_copy):
    _edit(
        config_copy / "action_ladder.yaml",
        lambda d: d["levels"]["L3"].update(requires_approval=False),
    )
    with pytest.raises(ValidationError, match="L3 must require human approval"):
        load_settings(config_copy, env={})


def test_threshold_min_above_max_is_rejected(config_copy):
    _edit(config_copy / "thresholds.yaml", lambda d: d["ph"].update(min=9, max=8))
    with pytest.raises(ValidationError, match="min must be below max"):
        load_settings(config_copy, env={})


def test_banned_names_include_synonyms(settings):
    names = settings.banned.all_names()
    assert "chloramphenicol" in names
    assert "furazolidone" in names


def test_defaults_are_offline_mock(settings):
    env = settings.runtime
    assert env.provider == "mock"
    assert env.tracing == "jsonl"
    assert settings.fault_injection == "none"


def test_env_overrides():
    s = load_settings(
        env={
            "LLM_PROVIDER": "lmstudio",
            "LMSTUDIO_AGENT_MODEL": "qwen2.5-7b-instruct",
            "LMSTUDIO_JUDGE_MODEL": "",
            "PONDGUARD_INJECT_FAULT": "uncited_step",
            "DATA_DIR": "./data",
        }
    )
    assert s.runtime.provider == "lmstudio"
    assert s.runtime.lmstudio_agent_model == "qwen2.5-7b-instruct"
    assert s.runtime.lmstudio_judge_model is None  # blank means unset
    assert s.fault_injection == "uncited_step"
    assert s.runtime.data_dir == (REPO_ROOT / "data").resolve()


def test_unknown_fault_or_provider_is_rejected():
    with pytest.raises(ValidationError):
        load_settings(env={"PONDGUARD_INJECT_FAULT": "drop_everything"})
    with pytest.raises(ValidationError):
        RuntimeEnv.from_env({"LLM_PROVIDER": "groq"})
