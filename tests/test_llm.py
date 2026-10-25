"""The LLM layer: mock outputs, roles, token accounting and the LM Studio wiring."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from pondguard.config import LLMSettings, load_settings
from pondguard.identity import model_identity, parse_selected_runtime
from pondguard.llm import (
    JUDGE_TASKS,
    TASK_SCHEMAS,
    LLMClient,
    LLMConfigError,
    PassageGrades,
    RankedHypotheses,
    TokenBudget,
    TokenCapExceeded,
    fallback_response,
    get_llm,
    mock_response,
)
from pondguard.llm_cache import LLMCache, cache_key

LMSTUDIO_ENV = {
    "LLM_PROVIDER": "lmstudio",
    "LMSTUDIO_AGENT_MODEL": "qwen2.5-7b-instruct",
    "LMSTUDIO_JUDGE_MODEL": "meta-llama-3.1-8b-instruct",
}


@pytest.fixture
def mock_settings():
    return load_settings(env={})


@pytest.mark.parametrize("task", sorted(TASK_SCHEMAS))
def test_mock_returns_schema_valid_output_for_every_task(mock_settings, task):
    role = "judge" if task in JUDGE_TASKS else "agent"
    client = get_llm(role, mock_settings)
    result = client.run(task, "prompt text", context={"passage_ids": ["kb-1", "kb-2"]})
    schema = TASK_SCHEMAS[task]
    assert isinstance(result.output, schema)
    schema.model_validate(result.output.model_dump())  # round-trips through validation
    assert result.usage.estimated and result.usage.total > 0


def test_mock_outputs_only_cite_given_passages():
    ranked = mock_response("rank", {"passage_ids": ["kb-9"]})
    assert ranked.hypotheses[0].citations == ["kb-9"]
    assert mock_response("rank", {}).hypotheses == []
    grades = mock_response("grade", {"passage_ids": ["a", "b"]})
    assert [g.passage_id for g in grades.grades] == ["a", "b"]


def test_tasks_are_locked_to_their_role(mock_settings):
    with pytest.raises(LLMConfigError, match="judge"):
        get_llm("agent", mock_settings).run("check_claim", "p")
    with pytest.raises(LLMConfigError, match="agent"):
        get_llm("judge", mock_settings).run("plan", "p")


def test_token_budget_refuses_calls_after_the_cap(mock_settings):
    client = get_llm("agent", mock_settings, budget=TokenBudget(cap=30))
    client.run("plan", "x" * 100)  # ~25 input tokens + output, crosses the cap
    assert client.budget.used >= 30
    with pytest.raises(TokenCapExceeded):
        client.run("summarise", "more")
    assert client.budget.hit
    assert client.budget.calls == 1


def test_lmstudio_client_points_at_local_server():
    settings = load_settings(env=LMSTUDIO_ENV)
    agent = get_llm("agent", settings)
    assert agent.provider == "lmstudio"
    assert agent.model == "qwen2.5-7b-instruct"
    assert str(agent.chat_model.openai_api_base).startswith("http://localhost:1234/v1")
    assert agent.chat_model.temperature == 0
    assert get_llm("judge", settings).model == "meta-llama-3.1-8b-instruct"


def test_judge_must_differ_from_agent():
    same = {**LMSTUDIO_ENV, "LMSTUDIO_JUDGE_MODEL": "qwen2.5-7b-instruct"}
    with pytest.raises(LLMConfigError, match="differ"):
        get_llm("judge", load_settings(env=same))


def test_missing_model_key_is_a_clear_error():
    with pytest.raises(LLMConfigError, match="LMSTUDIO_AGENT_MODEL"):
        get_llm("agent", load_settings(env={"LLM_PROVIDER": "lmstudio"}))


class _FakeChat:
    """Stands in for ChatOpenAI: returns what with_structured_output(include_raw=True) would.

    `responses` are served in order; an Exception instance is raised instead of returned.
    Every prompt it receives is kept in `.prompts`.
    """

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def with_structured_output(self, schema, **kwargs):
        assert kwargs["method"] == "json_schema" and kwargs["include_raw"] is True

        def respond(prompt):
            self.prompts.append(prompt)
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        return RunnableLambda(respond)


def _chat_client(*responses, cache=None, repeat=0) -> LLMClient:
    return LLMClient(
        role="agent",
        provider="lmstudio",
        model="fake",
        budget=TokenBudget(cap=1000),
        chat_model=_FakeChat(*responses),
        settings=LLMSettings(temperature=0, request_timeout_s=5, retry_backoff_s=0),
        cache=cache,
        repeat=repeat,
    )


def _ok(grades=(), tokens=(120, 30)):
    raw = AIMessage(
        content="{}",
        usage_metadata={
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
    )
    return {"raw": raw, "parsed": PassageGrades(grades=list(grades)), "parsing_error": None}


def _bad(text="not json"):
    return {"raw": AIMessage(content=text), "parsed": None, "parsing_error": ValueError("bad")}


def test_real_provider_usage_comes_from_the_model_response():
    client = _chat_client(_ok())
    result = client.run("grade", "prompt")
    assert (result.usage.input_tokens, result.usage.output_tokens) == (120, 30)
    assert not result.usage.estimated
    assert client.budget.used == 150
    assert (result.attempts, result.repaired, result.fallback) == (1, False, False)


# --- Phase 2c: schema repair and fallback ------------------------------------


def test_invalid_output_is_repaired_by_one_re_ask():
    client = _chat_client(_bad("oops"), _ok())
    result = client.run("grade", "prompt")
    assert result.repaired and not result.fallback and result.attempts == 2
    # the re-ask shows the model its own output and the error
    second = client.chat_model.prompts[1]
    assert ("ai", "oops") in second
    assert "did not match the required JSON schema" in second[-1][1]
    assert client.budget.used > 150  # the failed attempt's tokens are charged too


def test_validation_error_raised_by_the_client_is_repaired_too():
    # Real case from the smoke test: Qwen returned confidence=85 (a percentage) and the
    # OpenAI client raised pydantic's ValidationError itself instead of returning it.
    try:
        RankedHypotheses.model_validate(
            {"hypotheses": [{"condition": "ehp", "confidence": 85, "citations": ["kb-1"]}]}
        )
    except ValidationError as exc:
        client_error = exc
    fixed = {
        "raw": AIMessage(content="{}"),
        "parsed": RankedHypotheses(hypotheses=[]),
        "parsing_error": None,
    }
    client = _chat_client(client_error, fixed)
    result = client.run("rank", "prompt")
    assert result.repaired and not result.fallback
    assert "less than or equal to 1" in client.chat_model.prompts[1][-1][1]


def test_still_invalid_after_repair_falls_back_to_a_safe_default():
    client = _chat_client(_bad(), _bad())
    result = client.run("grade", "prompt", context={"passage_ids": ["kb-1", "kb-2"]})
    assert result.fallback and result.attempts == 2
    assert [g.grade for g in result.output.grades] == [0.0, 0.0]  # nothing counts as relevant
    assert result.error


@pytest.mark.parametrize("task", sorted(TASK_SCHEMAS))
def test_every_fallback_is_schema_valid_and_cautious(task):
    out = fallback_response(task, {"passage_ids": ["kb-1"], "query": "white strings"})
    TASK_SCHEMAS[task].model_validate(out.model_dump())
    if task == "rank":
        assert out.hypotheses == []  # -> Diagnostician abstains
    if task == "check_claim":
        assert out.supported is False
    if task == "advise":
        assert all(not s.citations for s in out.steps)  # -> Critic flags it for a human


# --- Phase 2c: retry ------------------------------------------------------------


def test_connection_errors_are_retried_then_succeed():
    client = _chat_client(ConnectionError("server starting"), ConnectionError("still"), _ok())
    result = client.run("grade", "prompt")
    assert result.attempts == 1 and not result.fallback
    assert len(client.chat_model.prompts) == 3


def test_retries_are_bounded():
    errors = [ConnectionError("down")] * 4  # max_retries=3 -> 4 tries in total
    client = _chat_client(*errors)
    with pytest.raises(ConnectionError):
        client.run("grade", "prompt")
    assert len(client.chat_model.prompts) == 4


def test_other_errors_are_not_retried():
    client = _chat_client(KeyError("bug"), _ok())
    with pytest.raises(KeyError):
        client.run("grade", "prompt")
    assert len(client.chat_model.prompts) == 1


# --- Phase 2c: cache ------------------------------------------------------------


def test_cache_hit_returns_original_tokens_and_latency(tmp_path):
    cache = LLMCache(tmp_path / "llm_cache.db")
    first = _chat_client(_ok(tokens=(200, 50)), cache=cache).run("grade", "same prompt")
    second_client = _chat_client(cache=cache)  # no responses queued: must not call the model
    second = second_client.run("grade", "same prompt")
    assert second.cache_hit and not first.cache_hit
    assert second.usage.total == 250  # cost is the original call's, never zero
    assert second.latency_s == pytest.approx(first.latency_s)
    assert second_client.chat_model.prompts == []
    assert second_client.budget.used == 250


def test_different_prompt_or_repeat_misses_the_cache(tmp_path):
    cache = LLMCache(tmp_path / "llm_cache.db")
    _chat_client(_ok(), cache=cache).run("grade", "prompt A")
    assert not _chat_client(_ok(), cache=cache).run("grade", "prompt B").cache_hit
    assert not _chat_client(_ok(), cache=cache, repeat=1).run("grade", "prompt A").cache_hit
    assert len(cache) == 3


def test_fallbacks_are_never_cached(tmp_path):
    cache = LLMCache(tmp_path / "llm_cache.db")
    _chat_client(_bad(), _bad(), cache=cache).run("grade", "prompt")
    assert len(cache) == 0


def test_no_cache_switch_and_mock_mode(tmp_path):
    env = {**LMSTUDIO_ENV, "DATA_DIR": str(tmp_path)}
    assert get_llm("agent", load_settings(env=env)).cache is not None
    assert get_llm("agent", load_settings(env=env), use_cache=False).cache is None
    assert get_llm("agent", load_settings(env={"DATA_DIR": str(tmp_path)})).cache is None


def test_cache_key_covers_model_task_prompt_and_repeat():
    base = dict(provider="lmstudio", model="m", task="grade", prompt="p", temperature=0, repeat=0)
    key = cache_key(**base)
    for change in ({"model": "m2"}, {"task": "rank"}, {"prompt": "q"}, {"repeat": 1}):
        assert cache_key(**{**base, **change}) != key
    assert cache_key(**base) == key


# --- Phase 2c: model identity -----------------------------------------------------


def test_selected_runtime_is_read_from_lms_table():
    table = (
        "LLM ENGINE                          SELECTED    MODEL FORMAT\n"
        "llama.cpp-win-x86_64-avx2@2.31.2                GGUF\n"
        "llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.64.0  ✓   GGUF\n"
    )
    assert parse_selected_runtime(table) == "llama.cpp-win-x86_64-nvidia-cuda12-avx2@2.64.0"
    assert parse_selected_runtime("no engines") is None


def test_identity_for_non_lmstudio_providers_is_not_invented():
    ident = model_identity("mock", "mock-agent")
    assert ident.as_dict() == {
        "provider": "mock",
        "model_key": "mock-agent",
        "file": None,
        "quantization": None,
        "params": None,
        "lmstudio_version": None,
        "runtime": None,
    }
