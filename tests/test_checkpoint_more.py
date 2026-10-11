# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Checkpointer cases beyond LangGraph's conformance suite: odd ids, concurrency, cleanup, shared state."""

import concurrent.futures as cf
import operator
from typing import Annotated, TypedDict

import hotkv
import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.hotkv import AsyncHotKVSaver, HotKVSaver
from langgraph.graph import END, START, StateGraph

from conftest import URL, wipe


def cfg(thread, ns="", cid=None):
    c = {"thread_id": thread, "checkpoint_ns": ns}
    if cid:
        c["checkpoint_id"] = cid
    return {"configurable": c}


def put_one(saver, thread, ns="", parent=None, value=1):
    cp = empty_checkpoint()
    cp["channel_values"] = {"v": value}
    cp["channel_versions"] = {"v": saver.get_next_version(None, None)}
    config = cfg(thread, ns, parent)
    return saver.put(config, cp, {"source": "loop", "step": 0, "parents": {}}, dict(cp["channel_versions"])), cp


@pytest.fixture
def saver(prefix):
    with HotKVSaver.from_conn_string(URL, prefix=prefix) as s:
        yield s


@pytest.mark.parametrize("thread", ["plain", "with:colon", "with space", "path/like", "日本語", "emoji-\U0001F600", "a" * 500])
def test_odd_thread_ids_roundtrip_and_stay_separate(saver, thread):
    saved, cp = put_one(saver, thread, value=thread)
    got = saver.get_tuple(saved)
    assert got.checkpoint["channel_values"] == {"v": thread}
    assert [t.checkpoint["id"] for t in saver.list(cfg(thread))] == [cp["id"]]
    saver.delete_thread(thread)
    assert saver.get_tuple(cfg(thread)) is None


def test_threads_that_share_a_prefix_do_not_mix(saver):
    put_one(saver, "t1", value="one")
    put_one(saver, "t10", value="ten")
    assert [t.checkpoint["channel_values"]["v"] for t in saver.list(cfg("t1"))] == ["one"]
    saver.delete_thread("t1")
    assert saver.get_tuple(cfg("t10")).checkpoint["channel_values"] == {"v": "ten"}


def test_control_characters_in_ids_are_refused(saver):
    with pytest.raises(ValueError):
        put_one(saver, "bad" + chr(31) + "id")
    with pytest.raises(ValueError):
        put_one(saver, "ok", ns="bad" + chr(31) + "ns")


def test_list_across_all_threads_sees_every_thread(saver):
    for i in range(5):
        put_one(saver, f"all{i}")
    seen = {t.config["configurable"]["thread_id"] for t in saver.list(None)}
    assert {f"all{i}" for i in range(5)} <= seen


def test_many_threads_written_concurrently(prefix):
    with HotKVSaver.from_conn_string(URL, prefix=prefix) as s:
        def work(i):
            parent = None
            for step in range(5):
                saved, _ = put_one(s, f"cc{i}", parent=parent, value=f"{i}-{step}")
                s.put_writes(saved, [("v", step)], task_id=f"task{step}")
                parent = saved["configurable"]["checkpoint_id"]
            return i

        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            assert sorted(pool.map(work, range(24))) == list(range(24))
        for i in range(24):
            history = list(s.list(cfg(f"cc{i}")))
            assert len(history) == 5
            assert history[0].checkpoint["channel_values"] == {"v": f"{i}-4"}
            assert history[0].pending_writes == [("task4", "v", 4)]


def test_two_savers_share_one_thread(prefix):
    with HotKVSaver.from_conn_string(URL, prefix=prefix) as a, HotKVSaver.from_conn_string(URL, prefix=prefix) as b:
        saved, cp = put_one(a, "shared", value="from-a")
        assert b.get_tuple(saved).checkpoint["channel_values"] == {"v": "from-a"}
        b.put_writes(saved, [("v", "write-from-b")], task_id="tb")
        assert a.get_tuple(saved).pending_writes == [("tb", "v", "write-from-b")]


def test_repeating_a_non_special_write_keeps_the_first(saver):
    saved, _ = put_one(saver, "w")
    saver.put_writes(saved, [("v", "first")], task_id="t")
    saver.put_writes(saved, [("v", "second")], task_id="t")
    assert saver.get_tuple(saved).pending_writes == [("t", "v", "first")]


def test_blobs_shared_between_checkpoints_survive_pruning_one_of_them(saver):
    first, _ = put_one(saver, "blob", value="kept")
    # a second checkpoint that reuses the same channel version (the channel did not change)
    cp = empty_checkpoint()
    cp["channel_values"] = {"v": "kept"}
    version = saver.get_tuple(first).checkpoint["channel_versions"]["v"]
    cp["channel_versions"] = {"v": version}
    second = saver.put(first, cp, {"source": "loop", "step": 1, "parents": {}}, {})
    saver.prune(["blob"])
    latest = saver.get_tuple(cfg("blob"))
    assert latest.checkpoint["id"] == cp["id"]
    assert latest.checkpoint["channel_values"] == {"v": "kept"}, "a value the kept checkpoint still uses was deleted"


def test_delete_for_runs_only_touches_those_runs(saver):
    one, _ = put_one(saver, "runs")
    cp = empty_checkpoint()
    cp["channel_values"] = {"v": 2}
    cp["channel_versions"] = {"v": saver.get_next_version(None, None)}
    saver.put(one, cp, {"source": "loop", "step": 1, "parents": {}, "run_id": "run-b"}, dict(cp["channel_versions"]))
    saver.delete_for_runs(["run-b"])
    ids = [t.checkpoint["id"] for t in saver.list(cfg("runs"))]
    assert ids == [one["configurable"]["checkpoint_id"]]


def test_copy_thread_is_independent_of_the_source(saver):
    saved, _ = put_one(saver, "src", value="orig")
    saver.put_writes(saved, [("v", "w")], task_id="t")
    saver.copy_thread("src", "dst")
    saver.delete_thread("src")
    got = saver.get_tuple(cfg("dst"))
    assert got.checkpoint["channel_values"] == {"v": "orig"} and got.pending_writes == [("t", "v", "w")]


def test_every_thread_operation_leaves_no_keys_behind(prefix):
    with HotKVSaver.from_conn_string(URL, prefix=prefix) as s:
        saved, _ = put_one(s, "gone")
        s.put_writes(saved, [("v", 1)], task_id="t")
        put_one(s, "child", ns="sub:1|inner:2")
        s.copy_thread("gone", "gone2")
        for t in ("gone", "gone2", "child"):
            s.delete_thread(t)
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
        wipe(client, prefix)
        client.close()


def test_prune_and_delete_are_safe_to_repeat(saver):
    put_one(saver, "rep")
    saver.prune(["rep"])
    saver.prune(["rep"])
    saver.prune(["rep"], strategy="delete")
    saver.prune(["rep"], strategy="delete")
    saver.delete_thread("rep")
    saver.delete_thread("never-existed")
    saver.delete_for_runs(["no-such-run"])
    assert saver.get_tuple(cfg("rep")) is None


def test_a_graph_runs_on_threads_in_parallel(prefix):
    class S(TypedDict):
        log: Annotated[list, operator.add]

    g = StateGraph(S)
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    with HotKVSaver.from_conn_string(URL, prefix=prefix) as s:
        graph = g.compile(checkpointer=s)

        def run(i):
            return graph.invoke({"log": [str(i)]}, {"configurable": {"thread_id": f"par{i}"}})["log"]

        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run, range(32)))
        assert results == [[str(i), "a", "b"] for i in range(32)]


@pytest.mark.asyncio
async def test_async_saver_odd_ids_and_cleanup(prefix):
    async with AsyncHotKVSaver.from_conn_string(URL, prefix=prefix) as s:
        for thread in ("a:b", "日本", "x y"):
            cp = empty_checkpoint()
            cp["channel_values"] = {"v": thread}
            cp["channel_versions"] = {"v": 1}
            saved = await s.aput(cfg(thread), cp, {"source": "loop", "step": 0, "parents": {}}, {"v": 1})
            got = await s.aget_tuple(saved)
            assert got.checkpoint["channel_values"] == {"v": thread}
            await s.adelete_thread(thread)
            assert await s.aget_tuple(cfg(thread)) is None
