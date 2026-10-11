# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Concurrency, cross-thread use and the cleanup paths that run next to writers."""

import asyncio
import concurrent.futures as cf
import threading
import time
from typing import TypedDict

import hotkv
import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.hotkv import AsyncHotKVSaver, HotKVSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.hotkv import AsyncHotKVStore, HotKVStore

from conftest import URL
from test_store import DIMS, INDEX, embed_many


def cfg(thread, cid=None):
    c = {"thread_id": thread, "checkpoint_ns": ""}
    if cid:
        c["checkpoint_id"] = cid
    return {"configurable": c}


def checkpoint(value, version):
    cp = empty_checkpoint()
    cp["channel_values"] = {"v": value}
    cp["channel_versions"] = {"v": version}
    return cp


META = {"source": "loop", "step": 0, "parents": {}}


def test_pruning_while_writing_never_loses_a_value_a_new_checkpoint_reuses(prefix):
    """New checkpoints keep pointing at an old channel value; a pruner must not delete it from under them."""
    with HotKVSaver.from_conn_string(URL, prefix=prefix) as s:
        first = s.put(cfg("t"), checkpoint("shared", 1), META, {"v": 1})
        stop = threading.Event()
        problems = []

        def pruner():
            while not stop.is_set():
                s.prune(["t"])

        def writer():
            parent = first
            for _ in range(60):
                # version 1 again: the value is not rewritten, the checkpoint only refers to it
                saved = s.put(parent, checkpoint("shared", 1), META, {})
                got = s.get_tuple(saved)
                if got is None or got.checkpoint["channel_values"] != {"v": "shared"}:
                    problems.append(got.checkpoint["channel_values"] if got else None)
                parent = saved

        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            p = pool.submit(pruner)
            w = pool.submit(writer)
            w.result()
            stop.set()
            p.result()
        assert problems == [], f"{len(problems)} checkpoints lost their value"
        assert s.get_tuple(cfg("t")).checkpoint["channel_values"] == {"v": "shared"}


def test_a_checkpoint_written_while_a_thread_is_deleted_stays_listed(prefix):
    with HotKVSaver.from_conn_string(URL, prefix=prefix) as s:
        stop = threading.Event()

        def deleter():
            while not stop.is_set():
                s.delete_thread("t")

        def writer():
            for i in range(80):
                s.put(cfg("t"), checkpoint(i, 1), META, {"v": 1})

        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            d = pool.submit(deleter)
            pool.submit(writer).result()
            stop.set()
            d.result()
        # after the dust settles, whatever exists must be reachable
        s.put(cfg("t"), checkpoint("last", 1), META, {"v": 1})
        client = hotkv.from_url(URL)
        try:
            stored = {i.decode() for i in client.zrangebylex(f"{prefix}:ix:t" + chr(31), "-", "+")}
        finally:
            client.close()
        listed = {t.checkpoint["id"] for t in s.list(cfg("t"))}
        assert stored <= listed, "checkpoints exist that list() cannot see"
        assert any(t.checkpoint["channel_values"] == {"v": "last"} for t in s.list(cfg("t")))
        s.delete_thread("t")


@pytest.mark.asyncio
async def test_sync_calls_from_other_threads_work_on_the_async_saver(prefix):
    async with AsyncHotKVSaver.from_conn_string(URL, prefix=prefix) as s:
        saved = await s.aput(cfg("a"), checkpoint("x", 1), META, {"v": 1})
        # what LangGraph does for a sync node: calls the sync API from a worker thread
        got = await asyncio.to_thread(s.get_tuple, saved)
        assert got.checkpoint["channel_values"] == {"v": "x"}
        listed = await asyncio.to_thread(lambda: list(s.list(cfg("a"))))
        assert len(listed) == 1
        await asyncio.to_thread(s.put_writes, saved, [("v", 1)], "task")
        await asyncio.to_thread(s.delete_thread, "a")
        assert await s.aget_tuple(cfg("a")) is None
        with pytest.raises(asyncio.InvalidStateError):
            s.get_tuple(saved)  # a blocking call on the event loop's own thread would deadlock


@pytest.mark.asyncio
async def test_a_sync_node_in_an_async_graph_can_use_the_store_and_the_saver(prefix):
    class S(TypedDict):
        x: int

    def remember(state, *, store):
        store.put(("m",), "k", {"v": state["x"]})
        return {"x": store.get(("m",), "k").value["v"] + 1}

    g = StateGraph(S)
    g.add_node("remember", remember)
    g.add_edge(START, "remember")
    g.add_edge("remember", END)
    async with AsyncHotKVStore.from_conn_string(URL, prefix=prefix + "s") as store:
        async with AsyncHotKVSaver.from_conn_string(URL, prefix=prefix) as saver:
            graph = g.compile(checkpointer=saver, store=store)
            out = await graph.ainvoke({"x": 1}, {"configurable": {"thread_id": "g"}})
            assert out == {"x": 2}
            assert (await graph.aget_state({"configurable": {"thread_id": "g"}})).values == {"x": 2}
            assert (await store.aget(("m",), "k")).value == {"v": 1}
    client = hotkv.from_url(URL)
    try:
        from conftest import wipe

        wipe(client, prefix + "s")
    finally:
        client.close()


def test_a_read_cannot_undo_a_concurrent_change_to_the_ttl(prefix):
    """The refresh after a read is skipped when a write changed the item's ttl in between."""
    from langgraph.store.hotkv._core import _REFRESH_SCRIPT

    client = hotkv.from_url(URL)
    try:
        key = f"{prefix}:it:probe"
        client.hset(key, {"ttl": "5.0", "v": "{}"})
        client.persist(key)
        assert client.execute_command("EVAL", _REFRESH_SCRIPT, 1, key, "9.0", 540) == 0  # ttl changed meanwhile
        assert client.ttl(key) == -1, "the item was given an expiry although its ttl had changed"
        assert client.execute_command("EVAL", _REFRESH_SCRIPT, 1, key, "5.0", 300) == 1
        assert 0 < client.ttl(key) <= 300
    finally:
        client.delete(f"{prefix}:it:probe")
        client.close()


def test_forgetting_an_expired_key_does_not_remove_an_item_written_since(prefix):
    from langgraph.store.hotkv._core import _DROP_SCRIPT

    client = hotkv.from_url(URL)
    try:
        keys, item = f"{prefix}:nk:n", f"{prefix}:it:n" + chr(31) + "k"
        client.sadd(keys, "k")
        client.hset(item, {"v": "{}"})
        assert client.execute_command("EVAL", _DROP_SCRIPT, 2, keys, item, "k") == 0
        assert client.sismember(keys, "k")
        client.delete(item)
        assert client.execute_command("EVAL", _DROP_SCRIPT, 2, keys, item, "k") == 1
        assert not client.sismember(keys, "k")
    finally:
        client.delete(keys, item)
        client.close()


def test_old_vectors_of_an_expired_item_are_not_returned_for_its_new_version(prefix):
    index = {"dims": DIMS, "embed": embed_many, "fields": ["tags[*]"]}
    with HotKVStore.from_conn_string(URL, prefix=prefix, index=index) as s:
        s.put(("kb",), "d", {"tags": ["red apples", "blue whales"]}, ttl=0.01)  # 1 s
        time.sleep(1.8)
        s.put(("kb",), "d", {"tags": ["red apples"]})  # the expired item's record is gone, so its vectors were forgotten
        hits = s.search(("kb",), query="blue whales", limit=5)
        assert not hits or hits[0].score < 0.5, "a vector of the expired version still answers"


def test_the_store_handles_many_threads_putting_and_searching(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix, index=INDEX) as s:
        def work(i):
            s.put(("kb", f"u{i % 4}"), f"k{i}", {"text": f"note about topic number {chr(97 + i % 20)}{chr(97 + i % 7)}"})
            return len(s.search(("kb",), query="note about topic", limit=5))

        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            counts = list(pool.map(work, range(60)))
        assert all(c >= 1 for c in counts)
        assert len(s.search(("kb",), limit=100)) == 60
