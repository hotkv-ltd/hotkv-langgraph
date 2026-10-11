# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Real LangGraph graphs running on the HotKV checkpointers."""

import operator
import sys
from typing import Annotated, TypedDict

import hotkv
import hotkv.asyncio
import pytest
from langgraph.checkpoint.hotkv import AsyncHotKVSaver, HotKVSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from conftest import URL


class State(TypedDict):
    log: Annotated[list, operator.add]
    n: int


def step_a(state: State):
    return {"log": ["a"], "n": state.get("n", 0) + 1}


def step_b(state: State):
    return {"log": ["b"], "n": state["n"] + 10}


def build(checkpointer, *, with_interrupt=False):
    g = StateGraph(State)
    g.add_node("a", step_a)
    if with_interrupt:
        def ask(state: State):
            answer = interrupt("proceed?")
            return {"log": [f"answer={answer}"]}

        g.add_node("ask", ask)
        g.add_node("b", step_b)
        g.add_edge(START, "a")
        g.add_edge("a", "ask")
        g.add_edge("ask", "b")
    else:
        g.add_node("b", step_b)
        g.add_edge(START, "a")
        g.add_edge("a", "b")
    g.add_edge("b", END)
    return g.compile(checkpointer=checkpointer)


def cfg(thread):
    return {"configurable": {"thread_id": thread}}


@pytest.fixture
def saver(prefix):
    with HotKVSaver.from_conn_string(URL, prefix=prefix) as s:
        yield s


def test_state_survives_a_new_saver_instance(saver, prefix):
    graph = build(saver)
    out = graph.invoke({"log": [], "n": 0}, cfg("t1"))
    assert out == {"log": ["a", "b"], "n": 11}
    with HotKVSaver.from_conn_string(URL, prefix=prefix) as other:
        state = build(other).get_state(cfg("t1"))
        assert state.values == {"log": ["a", "b"], "n": 11}
        assert state.next == ()


def test_history_lists_every_step_newest_first_and_can_be_forked(saver):
    graph = build(saver)
    graph.invoke({"log": [], "n": 0}, cfg("t2"))
    history = list(graph.get_state_history(cfg("t2")))
    assert [s.metadata["step"] for s in history] == sorted([s.metadata["step"] for s in history], reverse=True)
    assert len(history) >= 4
    # go back to the checkpoint before node b ran and take another path
    before_b = next(s for s in history if s.next == ("b",))
    fork = graph.update_state(before_b.config, {"log": ["edited"]})
    out = graph.invoke(None, fork)
    assert out["log"] == ["a", "edited", "b"]
    # the original run is untouched
    assert graph.get_state(cfg("t2")).values["log"][-1] == "b"


def test_interrupt_and_resume(saver):
    graph = build(saver, with_interrupt=True)
    first = graph.invoke({"log": [], "n": 0}, cfg("t3"))
    assert first["__interrupt__"][0].value == "proceed?"
    assert graph.get_state(cfg("t3")).next == ("ask",)
    done = graph.invoke(Command(resume="yes"), cfg("t3"))
    assert done["log"] == ["a", "answer=yes", "b"]


def test_a_second_run_on_the_same_thread_continues_its_state(saver):
    graph = build(saver)
    graph.invoke({"log": [], "n": 0}, cfg("t4"))
    out = graph.invoke({"log": ["again"]}, cfg("t4"))
    assert out["log"][:3] == ["a", "b", "again"]


def test_subgraph_checkpoints_use_their_own_namespace(saver):
    inner = StateGraph(State)
    inner.add_node("a", step_a)
    inner.add_edge(START, "a")
    inner.add_edge("a", END)
    outer = StateGraph(State)
    outer.add_node("sub", inner.compile())
    outer.add_node("b", step_b)
    outer.add_edge(START, "sub")
    outer.add_edge("sub", "b")
    outer.add_edge("b", END)
    graph = outer.compile(checkpointer=saver)
    out = graph.invoke({"log": [], "n": 0}, cfg("t5"))
    assert out == {"log": ["a", "b"], "n": 11}
    namespaces = {t.config["configurable"]["checkpoint_ns"] for t in saver.list(cfg("t5"))}
    assert "" in namespaces and any(ns.startswith("sub") for ns in namespaces)


def test_big_and_binary_values_roundtrip(saver):
    class Blob(TypedDict):
        data: bytes
        text: str

    g = StateGraph(Blob)
    g.add_node("n", lambda s: {"data": s["data"][::-1], "text": s["text"] + "é"})
    g.add_edge(START, "n")
    g.add_edge("n", END)
    graph = g.compile(checkpointer=saver)
    payload = bytes(range(256)) * 4096  # 1 MiB, every byte value
    graph.invoke({"data": payload, "text": "x" * 500_000}, cfg("t6"))
    state = graph.get_state(cfg("t6")).values
    assert state["data"] == payload[::-1]
    assert state["text"] == "x" * 500_000 + "é"


def test_delete_thread_removes_every_key(saver, prefix):
    graph = build(saver)
    graph.invoke({"log": [], "n": 0}, cfg("t7"))
    saver.delete_thread("t7")
    assert graph.get_state(cfg("t7")).values == {}
    client = hotkv.from_url(URL)
    try:
        cursor, left = 0, []
        while True:
            cursor, keys = client.scan(cursor, match=f"{prefix}:*", count=500)
            left += keys
            if cursor == 0:
                break
        assert left == [], left
    finally:
        client.close()


def test_prune_keep_latest_keeps_the_thread_usable(saver):
    graph = build(saver)
    graph.invoke({"log": [], "n": 0}, cfg("t8"))
    before = len(list(graph.get_state_history(cfg("t8"))))
    saver.prune(["t8"])
    after = list(graph.get_state_history(cfg("t8")))
    assert before > 1 and len(after) == 1
    out = graph.invoke({"log": ["more"]}, cfg("t8"))
    assert out["log"][:3] == ["a", "b", "more"]


def test_delta_channel_survives_pruning(saver):
    try:
        from langgraph.channels.delta import DeltaChannel
    except ImportError:
        pytest.skip("this langgraph has no DeltaChannel")

    def reducer(state, writes):
        out = list(state or [])
        for w in writes:
            out.extend(w if isinstance(w, list) else [w])
        return out

    class D(TypedDict):
        items: Annotated[list, DeltaChannel(reducer, snapshot_frequency=4)]

    g = StateGraph(D)
    for i in range(6):
        g.add_node(f"n{i}", (lambda i: lambda s: {"items": [f"v{i}"]})(i))
    g.add_edge(START, "n0")
    for i in range(5):
        g.add_edge(f"n{i}", f"n{i + 1}")
    g.add_edge("n5", END)
    graph = g.compile(checkpointer=saver)
    graph.invoke({"items": []}, cfg("t9"))
    want = graph.get_state(cfg("t9")).values["items"]
    assert want == [f"v{i}" for i in range(6)]
    saver.prune(["t9"])
    assert graph.get_state(cfg("t9")).values["items"] == want


@pytest.mark.skipif(sys.version_info < (3, 11), reason="LangGraph needs Python 3.11+ for interrupt() in async nodes")
@pytest.mark.asyncio
async def test_async_graph_interrupt_resume_and_history(prefix):
    async with AsyncHotKVSaver.from_conn_string(URL, prefix=prefix) as s:
        graph = build(s, with_interrupt=True)
        first = await graph.ainvoke({"log": [], "n": 0}, cfg("a1"))
        assert first["__interrupt__"][0].value == "proceed?"
        done = await graph.ainvoke(Command(resume="ok"), cfg("a1"))
        assert done["log"] == ["a", "answer=ok", "b"]
        history = [h async for h in graph.aget_state_history(cfg("a1"))]
        assert len(history) >= 4
        await s.adelete_thread("a1")
        assert (await graph.aget_state(cfg("a1"))).values == {}


@pytest.mark.asyncio
async def test_concurrent_threads_do_not_mix(prefix):
    import asyncio

    async with AsyncHotKVSaver.from_conn_string(URL, prefix=prefix) as s:
        graph = build(s)

        async def run(i):
            return await graph.ainvoke({"log": [f"t{i}"], "n": i}, cfg(f"c{i}"))

        results = await asyncio.gather(*(run(i) for i in range(20)))
        for i, out in enumerate(results):
            assert out["log"] == [f"t{i}", "a", "b"] and out["n"] == i + 11
