# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""The HotKV store for LangGraph, against a live licensed server."""

import asyncio
import hashlib
import math
import re
import time

import hotkv
import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import InvalidNamespaceError, MatchCondition
from langgraph.store.hotkv import AsyncHotKVStore, HotKVStore
from typing_extensions import TypedDict

from conftest import URL

DIMS = 64


def embed(text: str):
    """A bag-of-words hash embedding: texts that share words are close."""
    vec = [0.0] * DIMS
    for word in re.findall(r"[a-z]+", text.lower()):
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        vec[h % DIMS] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed_many(texts):
    return [embed(t) for t in texts]


async def aembed_many(texts):
    return [embed(t) for t in texts]


INDEX = {"dims": DIMS, "embed": embed_many}


@pytest.fixture
def store(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix) as s:
        yield s


@pytest.fixture
def vstore(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix, index=INDEX) as s:
        yield s


def test_put_get_update_delete(store):
    ns = ("users", "ada")
    assert store.get(ns, "prefs") is None
    store.put(ns, "prefs", {"theme": "dark", "n": 1})
    item = store.get(ns, "prefs")
    assert item.value == {"theme": "dark", "n": 1}
    assert item.namespace == ns and item.key == "prefs"
    created, updated = item.created_at, item.updated_at
    time.sleep(0.02)
    store.put(ns, "prefs", {"theme": "light"})
    again = store.get(ns, "prefs")
    assert again.value == {"theme": "light"}
    assert again.created_at == created and again.updated_at > updated
    store.delete(ns, "prefs")
    assert store.get(ns, "prefs") is None
    assert store.list_namespaces() == []


def test_search_filters_orders_and_pages(store):
    ns = ("docs", "a")
    for i in range(6):
        store.put(ns, f"k{i}", {"n": i, "kind": "even" if i % 2 == 0 else "odd"})
        time.sleep(0.005)
    store.put(("docs", "b"), "other", {"n": 100, "kind": "even"})
    store.put(("elsewhere",), "x", {"n": 7})

    everything = store.search(("docs",), limit=50)
    assert len(everything) == 7
    assert everything[0].key == "other"  # newest first
    assert all(r.score is None for r in everything)

    assert {r.key for r in store.search(ns, filter={"kind": "even"})} == {"k0", "k2", "k4"}
    assert {r.key for r in store.search(("docs",), filter={"n": {"$gt": 3}})} == {"k4", "k5", "other"}
    assert {r.key for r in store.search(ns, filter={"n": {"$gte": 2, "$lt": 5}})} == {"k2", "k3", "k4"}
    assert {r.key for r in store.search(ns, filter={"kind": {"$ne": "even"}})} == {"k1", "k3", "k5"}
    assert [r.key for r in store.search(ns, limit=2)] == ["k5", "k4"]
    assert [r.key for r in store.search(ns, limit=2, offset=2)] == ["k3", "k2"]
    assert store.search(("nothing",)) == []


def test_list_namespaces(store):
    for ns in [("a", "b", "c"), ("a", "b", "d"), ("a", "x"), ("z", "b", "c")]:
        store.put(ns, "k", {"v": 1})
    assert store.list_namespaces() == [("a", "b", "c"), ("a", "b", "d"), ("a", "x"), ("z", "b", "c")]
    assert store.list_namespaces(prefix=("a",)) == [("a", "b", "c"), ("a", "b", "d"), ("a", "x")]
    assert store.list_namespaces(suffix=("c",)) == [("a", "b", "c"), ("z", "b", "c")]
    assert store.list_namespaces(prefix=("*", "b")) == [("a", "b", "c"), ("a", "b", "d"), ("z", "b", "c")]
    assert store.list_namespaces(max_depth=2) == [("a", "b"), ("a", "x"), ("z", "b")]
    assert store.list_namespaces(max_depth=1, limit=1, offset=1) == [("z",)]
    both = store.list_namespaces(prefix=("a",), suffix=("d",))
    assert both == [("a", "b", "d")]


def test_invalid_namespaces_are_refused(store):
    with pytest.raises(InvalidNamespaceError):
        store.put((), "k", {"a": 1})
    with pytest.raises(InvalidNamespaceError):
        store.put(("a.b",), "k", {"a": 1})
    with pytest.raises(InvalidNamespaceError):
        store.put(("langgraph", "x"), "k", {"a": 1})
    with pytest.raises(InvalidNamespaceError):
        store.put(("a", ""), "k", {"a": 1})


def test_stores_with_different_prefixes_do_not_see_each_other(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix + "-1") as one, HotKVStore.from_conn_string(
        URL, prefix=prefix + "-2"
    ) as two:
        one.put(("n",), "k", {"who": "one"})
        assert two.get(("n",), "k") is None
        assert two.search(("n",)) == []
        one.delete(("n",), "k")
    client = hotkv.from_url(URL)
    try:
        from conftest import wipe

        wipe(client, prefix + "-1")
        wipe(client, prefix + "-2")
    finally:
        client.close()


def test_items_expire(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix) as s:
        s.put(("tmp",), "short", {"a": 1}, ttl=0.02)  # 1.2 s
        s.put(("tmp",), "long", {"a": 2})
        assert s.get(("tmp",), "short") is not None
        time.sleep(2.0)
        assert s.get(("tmp",), "short") is None
        assert [r.key for r in s.search(("tmp",))] == ["long"]
        s.delete(("tmp",), "long")
        assert s.list_namespaces() == []


def test_reading_refreshes_the_ttl(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix) as s:
        s.put(("tmp",), "k", {"a": 1}, ttl=0.04)  # 2.4 s
        for _ in range(3):
            time.sleep(1.2)
            assert s.get(("tmp",), "k") is not None  # each read pushes the expiry out
        time.sleep(3.0)
        assert s.get(("tmp",), "k") is None


def test_semantic_search_ranks_by_meaning(vstore):
    ns = ("kb",)
    vstore.put(ns, "cats", {"text": "cats purr and chase mice"})
    vstore.put(ns, "dogs", {"text": "dogs bark and chase balls"})
    vstore.put(ns, "cars", {"text": "engines need oil and fuel"})
    vstore.put(("other",), "cats2", {"text": "cats purr softly"})

    hits = vstore.search(ns, query="mice and cats", limit=3)
    assert hits[0].key == "cats"
    assert hits[0].score > hits[-1].score
    assert all(0.0 <= h.score <= 1.0001 for h in hits)
    assert vstore.search(ns, query="engines fuel oil", limit=1)[0].key == "cars"
    # the namespace prefix confines the search
    assert {h.key for h in vstore.search(("kb",), query="cats")} <= {"cats", "dogs", "cars"}
    # filters apply to the semantic hits
    assert [h.key for h in vstore.search(ns, query="chase", filter={"text": "dogs bark and chase balls"})] == ["dogs"]


def test_semantic_search_follows_updates_and_deletes(vstore):
    ns = ("kb",)
    vstore.put(ns, "doc", {"text": "alpha beta gamma"})
    assert vstore.search(ns, query="alpha beta")[0].key == "doc"
    vstore.put(ns, "doc", {"text": "delta epsilon zeta"})
    stale = vstore.search(ns, query="alpha beta gamma", limit=1)
    fresh = vstore.search(ns, query="delta epsilon zeta", limit=1)
    assert fresh[0].key == "doc" and fresh[0].score > (stale[0].score if stale else 0)
    vstore.delete(ns, "doc")
    assert vstore.search(ns, query="delta epsilon zeta") == []


def test_index_option_controls_what_is_embedded(vstore):
    ns = ("kb",)
    vstore.put(ns, "a", {"title": "quantum entanglement", "body": "cooking pasta"}, index=["title"])
    vstore.put(ns, "b", {"title": "x", "body": "cooking pasta"}, index=False)
    hits = vstore.search(ns, query="quantum entanglement")
    assert [h.key for h in hits] == ["a"]
    assert vstore.search(ns, query="cooking pasta")[0].score < 0.5  # "body" of a was not indexed
    assert vstore.get(ns, "b").value["title"] == "x"  # still readable


def test_nested_field_paths_index_each_element(prefix):
    index = {"dims": DIMS, "embed": embed_many, "fields": ["tags[*]"]}
    with HotKVStore.from_conn_string(URL, prefix=prefix, index=index) as s:
        s.put(("kb",), "d", {"tags": ["red apples", "blue whales"]})
        assert s.search(("kb",), query="blue whales", limit=1)[0].key == "d"
        assert s.search(("kb",), query="red apples", limit=1)[0].key == "d"


def test_wrong_embedding_size_is_refused(prefix):
    index = {"dims": DIMS + 1, "embed": embed_many}
    with HotKVStore.from_conn_string(URL, prefix=prefix, index=index) as s:
        with pytest.raises(ValueError, match="dimensions"):
            s.put(("kb",), "d", {"text": "hello"})


def test_graph_nodes_can_use_the_store(prefix):
    class S(TypedDict):
        seen: str

    def remember(state, *, store):
        store.put(("memories", "u1"), "last", {"seen": state["seen"]})
        return {}

    def recall(state, *, store):
        return {"seen": store.get(("memories", "u1"), "last").value["seen"] + "!"}

    g = StateGraph(S)
    g.add_node("remember", remember)
    g.add_node("recall", recall)
    g.add_edge(START, "remember")
    g.add_edge("remember", "recall")
    g.add_edge("recall", END)
    with HotKVStore.from_conn_string(URL, prefix=prefix) as s:
        out = g.compile(store=s).invoke({"seen": "hello"})
        assert out == {"seen": "hello!"}


def test_one_batch_mixes_operations(store):
    from langgraph.store.base import GetOp, ListNamespacesOp, PutOp, SearchOp

    results = store.batch([
        PutOp(("b",), "1", {"v": 1}),
        PutOp(("b",), "2", {"v": 2}),
        GetOp(("b",), "1"),
        GetOp(("b",), "missing"),
        SearchOp(("b",), filter={"v": 2}),
        ListNamespacesOp(),
    ])
    assert results[0] is None and results[1] is None
    assert results[2].value == {"v": 1} and results[3] is None
    assert [r.key for r in results[4]] == ["2"]
    assert results[5] == [("b",)]


@pytest.mark.asyncio
async def test_async_store_end_to_end(prefix):
    index = {"dims": DIMS, "embed": aembed_many}
    async with AsyncHotKVStore.from_conn_string(URL, prefix=prefix, index=index) as s:
        await s.aput(("kb",), "cats", {"text": "cats purr and chase mice"})
        await s.aput(("kb",), "cars", {"text": "engines need oil and fuel"})
        got = await s.aget(("kb",), "cats")
        assert got.value["text"].startswith("cats")
        hits = await s.asearch(("kb",), query="mice and cats", limit=2)
        assert hits[0].key == "cats"
        assert await s.alist_namespaces() == [("kb",)]
        await asyncio.gather(*(s.aput(("many",), f"k{i}", {"i": i}) for i in range(30)))
        assert len(await s.asearch(("many",), limit=100)) == 30
        await s.adelete(("kb",), "cats")
        assert await s.aget(("kb",), "cats") is None
        with pytest.raises(asyncio.InvalidStateError):  # a blocking call on the event loop's own thread
            s.get(("kb",), "cars")
