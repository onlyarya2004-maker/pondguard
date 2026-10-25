"""The LLM layer: mock outputs, roles, token accounting and the LM Studio wiring."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from pondguard.config import load_settings
from pondguard.llm import (
    JUDGE_TASKS,
    TASK_SCHEMAS,
    LLMClient,
    LLMConfigError,
    LLMOutputError,
    PassageGrades,
    TokenBudget,
    TokenCapExceeded,
    get_llm,
    mock_response,
)

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
    """Stands in for ChatOpenAI: returns what with_structured_output(include_raw=True) would."""

    def __init__(self, response):
        self.response = response

    def with_structured_output(self, schema, **kwargs):
        assert kwargs["method"] == "json_schema" and kwargs["include_raw"] is True
        return RunnableLambda(lambda _prompt: self.response)


def _chat_client(response) -> LLMClient:
    return LLMClient(
        role="agent",
        provider="lmstudio",
        model="fake",
        budget=TokenBudget(cap=1000),
        chat_model=_FakeChat(response),
    )


def test_real_provider_usage_comes_from_the_model_response():
    raw = AIMessage(
        content="{}",
        usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
    )
    client = _chat_client({"raw": raw, "parsed": PassageGrades(grades=[]), "parsing_error": None})
    result = client.run("grade", "prompt")
    assert (result.usage.input_tokens, result.usage.output_tokens) == (120, 30)
    assert not result.usage.estimated
    assert client.budget.used == 150


def test_invalid_model_output_raises():
    raw = AIMessage(content="not json")
    client = _chat_client({"raw": raw, "parsed": None, "parsing_error": ValueError("bad")})
    with pytest.raises(LLMOutputError):
        client.run("grade", "prompt")
    assert client.budget.calls == 0  # nothing charged for a failed parse
