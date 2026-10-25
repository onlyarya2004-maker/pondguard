"""The graph state's event log appends across nodes instead of overwriting."""

from langgraph.graph import END, START, StateGraph

from pondguard.state import GraphState, event


def test_events_accumulate_across_nodes():
    def first(state: GraphState) -> dict:
        return {"events": [event("supervisor", "planned")], "step_count": 1}

    def second(state: GraphState) -> dict:
        return {"events": [event("monitor", "2 anomalies", count=2)], "step_count": 2}

    graph = StateGraph(GraphState)
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    out = graph.compile().invoke({"run_id": "r1", "events": []})
    assert [e["node"] for e in out["events"]] == ["supervisor", "monitor"]
    assert out["events"][1]["count"] == 2
    assert out["step_count"] == 2  # plain fields are overwritten, not appended
