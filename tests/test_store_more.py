# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Harder cases for the store: concurrency, other processes, updates, expiry and odd names."""

import concurrent.futures as cf
import time

import hotkv
import pytest
from langgraph.store.hotkv import AsyncHotKVStore, HotKVStore

from conftest import URL, wipe
from test_store import DIMS, INDEX, aembed_many, embed_many


def unique(i):
    """A word that differs per i (the test embedder only sees letters)."""
    return "".join(chr(97 + int(d)) for d in str(i)) + "zq"


@pytest.fixture
def vstore(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix, index=INDEX) as s:
        yield s


def test_updating_an_item_with_fewer_indexed_texts_removes_the_old_vectors(prefix):
    index = {"dims": DIMS, "embed": embed_many, "fields": ["tags[*]"]}
    with HotKVStore.from_conn_string(URL, prefix=prefix, index=index) as s:
        s.put(("kb",), "d", {"tags": ["red apples", "blue whales", "green frogs"]})
        assert s.search(("kb",), query="blue whales", limit=1)[0].score > 0.99
        s.put(("kb",), "d", {"tags": ["red apples"]})
        hits = s.search(("kb",), query="blue whales", limit=5)
        assert not hits or hits[0].score < 0.5, "a vector of the old version is still searchable"
        client = hotkv.from_url(URL)
        try:
            info = client.vsim_info(f"{prefix}:vi:kb")
            assert int(info["vectors"]) == 1
        finally:
            client.close()


def test_an_item_can_be_put_again_after_its_namespace_emptied(vstore):
    ns = ("kb",)
    vstore.put(ns, "a", {"text": "alpha beta"})
    vstore.delete(ns, "a")
    assert vstore.list_namespaces() == []
    vstore.put(ns, "b", {"text": "gamma delta"})
    assert vstore.search(ns, query="gamma delta", limit=1)[0].key == "b"
    assert vstore.list_namespaces() == [ns]


def test_a_second_process_dropping_the_index_does_not_break_writes(prefix):
    """Process A has written; process B deletes everything (dropping the index); A writes again."""
    with HotKVStore.from_conn_string(URL, prefix=prefix, index=INDEX) as a, HotKVStore.from_conn_string(
        URL, prefix=prefix, index=INDEX
    ) as b:
        a.put(("kb",), "one", {"text": "first document"})
        b.delete(("kb",), "one")
        a.put(("kb",), "two", {"text": "second document"})
        assert a.search(("kb",), query="second document", limit=1)[0].key == "two"
        assert b.get(("kb",), "two").value == {"text": "second document"}


def test_concurrent_writers_and_deleters_never_lose_an_item(prefix):
    """One thread keeps emptying a namespace while others add to it; every added item must survive."""
    with HotKVStore.from_conn_string(URL, prefix=prefix, index=INDEX) as s:
        stop = False

        def churn():
            n = 0
            while not stop:
                s.put(("kb",), "churn", {"text": f"churn {n}"})
                s.delete(("kb",), "churn")
                n += 1

        def add(i):
            s.put(("kb",), f"keep{i}", {"text": f"keeper {unique(i)} stays"})

        with cf.ThreadPoolExecutor(max_workers=6) as pool:
            churner = pool.submit(churn)
            time.sleep(0.05)
            list(pool.map(add, range(40)))
            stop = True
            churner.result()
        for i in range(40):
            assert s.get(("kb",), f"keep{i}") is not None, f"keep{i} is gone"
        assert {h.key for h in s.search(("kb",), limit=100) if h.key != "churn"} == {f"keep{i}" for i in range(40)}
        assert ("kb",) in s.list_namespaces()
        hit = s.search(("kb",), query=f"keeper {unique(7)} stays", limit=1)
        assert hit and hit[0].key == "keep7", "the vector index lost the items"


def test_get_can_leave_the_ttl_alone(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix) as s:
        s.put(("t",), "k", {"a": 1}, ttl=0.03)  # 1.8 s
        for _ in range(3):
            time.sleep(0.8)
            s.get(("t",), "k", refresh_ttl=False)
        assert s.get(("t",), "k") is None


def test_default_ttl_applies_and_a_plain_put_cannot_remove_it(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix, ttl={"default_ttl": 0.03}) as s:
        s.put(("t",), "k", {"a": 1})
        time.sleep(2.2)
        assert s.get(("t",), "k") is None


def test_putting_without_a_ttl_makes_an_item_permanent_again(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix) as s:
        s.put(("t",), "k", {"a": 1}, ttl=0.02)
        s.put(("t",), "k", {"a": 2})  # no ttl now
        time.sleep(1.6)
        assert s.get(("t",), "k").value == {"a": 2}


def test_unicode_and_punctuation_in_names(prefix):
    ns = ("users", "josé", "日本語")
    key = "k:with:colons and spaces/é"
    with HotKVStore.from_conn_string(URL, prefix=prefix) as s:
        s.put(ns, key, {"v": "☃"})
        assert s.get(ns, key).value == {"v": "☃"}
        assert [r.key for r in s.search(("users",))] == [key]
        assert s.list_namespaces(prefix=("users", "josé")) == [ns]
        s.delete(ns, key)
        assert s.get(ns, key) is None


def test_deleting_something_that_is_not_there_is_fine(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix) as s:
        s.delete(("nothing",), "nowhere")
        s.put(("a",), "k", {"v": 1})
        s.delete(("a",), "other")
        assert s.get(("a",), "k") is not None


def test_a_batch_with_repeated_puts_keeps_the_last(prefix):
    from langgraph.store.base import GetOp, PutOp

    with HotKVStore.from_conn_string(URL, prefix=prefix) as s:
        s.batch([PutOp(("b",), "k", {"v": 1}), PutOp(("b",), "k", {"v": 2}), PutOp(("b",), "k", {"v": 3})])
        assert s.batch([GetOp(("b",), "k")])[0].value == {"v": 3}


def test_semantic_offset_and_limit_page_through_results(vstore):
    for i in range(6):
        vstore.put(("kb",), f"d{i}", {"text": f"common words plus unique{i}"})
    first = [h.key for h in vstore.search(("kb",), query="common words", limit=3)]
    second = [h.key for h in vstore.search(("kb",), query="common words", limit=3, offset=3)]
    assert len(first) == 3 and len(second) == 3 and not set(first) & set(second)


def test_large_values_roundtrip(prefix):
    big = {"text": "x" * 2_000_000, "list": list(range(10_000))}
    with HotKVStore.from_conn_string(URL, prefix=prefix) as s:
        s.put(("big",), "k", big)
        assert s.get(("big",), "k").value == big


def test_values_must_be_json(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix) as s:
        with pytest.raises(TypeError):
            s.put(("a",), "k", {"v": object()})
        assert s.get(("a",), "k") is None


def test_filters_on_missing_fields_and_types(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix) as s:
        s.put(("f",), "a", {"n": 1})
        s.put(("f",), "b", {"n": "text"})
        s.put(("f",), "c", {"other": 1})
        assert {r.key for r in s.search(("f",), filter={"n": {"$gt": 0}})} == {"a"}
        assert {r.key for r in s.search(("f",), filter={"n": {"$ne": 1}})} == {"b", "c"}
        assert {r.key for r in s.search(("f",), filter={"n": "text"})} == {"b"}
        assert s.search(("f",), filter={"missing": 1}) == []


@pytest.mark.asyncio
async def test_async_store_ttl_and_concurrent_puts(prefix):
    async with AsyncHotKVStore.from_conn_string(URL, prefix=prefix) as s:
        import asyncio

        await asyncio.gather(*(s.aput(("c",), f"k{i}", {"i": i}) for i in range(50)))
        assert len(await s.asearch(("c",), limit=100)) == 50
        await s.aput(("c",), "short", {"i": -1}, ttl=0.02)  # 1.2 s, rounded up to 2 s on the server
        await asyncio.sleep(2.6)
        assert await s.aget(("c",), "short") is None


@pytest.mark.asyncio
async def test_async_semantic_updates(prefix):
    index = {"dims": DIMS, "embed": aembed_many}
    async with AsyncHotKVStore.from_conn_string(URL, prefix=prefix, index=index) as s:
        await s.aput(("kb",), "d", {"text": "first version about cats"})
        await s.aput(("kb",), "d", {"text": "second version about engines"})
        hit = (await s.asearch(("kb",), query="second version about engines", limit=1))[0]
        assert hit.key == "d" and hit.score > 0.8
        await s.adelete(("kb",), "d")
        assert await s.asearch(("kb",), query="second version about engines") == []


def test_cleanup_leaves_no_keys_or_indexes(prefix):
    with HotKVStore.from_conn_string(URL, prefix=prefix, index=INDEX) as s:
        s.put(("a", "b"), "k", {"text": "some text"})
        s.delete(("a", "b"), "k")
    client = hotkv.from_url(URL)
    try:
        cursor, left = 0, []
        while True:
            cursor, keys = client.scan(cursor, match=f"{prefix}:*", count=500)
            left += keys
            if cursor == 0:
                break
        assert left == [], left
        assert not [n for n in client.vsim_info() if n.decode().startswith(prefix)]
    finally:
        wipe(client, prefix)
        client.close()
