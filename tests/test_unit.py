# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Tests that need no server."""

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.base import InvalidNamespaceError, MatchCondition

from langgraph.checkpoint.hotkv._core import EMPTY, SEP, Core, Tx, _pack, _unpack, _write_order
from langgraph.store.hotkv._core import StoreCore
from langgraph.store.hotkv._core import SEP as STORE_SEP
from langgraph.store.hotkv._core import _compare, _namespace_matches, _passes, _validate


def test_pack_roundtrip_with_binary_parts():
    parts = [b"", b"abc", bytes(range(256)), b"\x00\x00"]
    assert _unpack(_pack(*parts)) == parts


def test_write_order_sorts_by_task_then_numeric_index():
    fields = [f"t2{SEP}1".encode(), f"t1{SEP}10".encode(), f"t1{SEP}2".encode()]
    assert [f.decode() for f in sorted(fields, key=_write_order)] == [f"t1{SEP}2", f"t1{SEP}10", f"t2{SEP}1"]


def test_put_writes_one_atomic_batch_with_the_expected_commands():
    core = Core(JsonPlusSerializer(), "pfx")
    cp = empty_checkpoint()
    cp["channel_values"] = {"a": 1}
    cp["channel_versions"] = {"a": 1, "b": 1}
    flow = core.put({"configurable": {"thread_id": "T", "checkpoint_ns": "", "checkpoint_id": "P"}}, cp, {"run_id": "R"}, {"a": 1, "b": 1})
    step = next(flow)
    assert isinstance(step, Tx)
    verbs = [c[0] for c in step]
    assert verbs.count("SET") == 2 and "HSET" in verbs and "ZADD" in verbs
    sets = {c[1]: c[2] for c in step if c[0] == "SET"}
    assert sets[f"pfx:bl:T{SEP}{SEP}b{SEP}1"] == EMPTY  # a channel with a version but no value
    assert ("SADD", f"pfx:run:R", f"T{SEP}{SEP}{cp['id']}") in step
    with pytest.raises(StopIteration) as done:
        flow.send([])
    assert done.value.value["configurable"]["checkpoint_id"] == cp["id"]


def test_thread_ids_with_the_separator_are_refused():
    core = Core(JsonPlusSerializer(), "pfx")
    with pytest.raises(ValueError):
        next(core.put({"configurable": {"thread_id": f"a{SEP}b"}}, empty_checkpoint(), {}, {}))


def test_unknown_prune_strategy_is_refused():
    with pytest.raises(ValueError):
        next(Core(JsonPlusSerializer(), "pfx").prune(["t"], "shred"))


def test_store_validates_namespaces_and_keys():
    _validate(("users", "ada"), "k")
    for bad in [(), ("a.b",), ("",), ("langgraph",), ("a\x1eb",)]:
        with pytest.raises(InvalidNamespaceError):
            _validate(bad)
    with pytest.raises(ValueError):
        _validate(("ok",), f"bad{STORE_SEP}key")


def test_filter_operators():
    assert _compare(5, {"$gt": 3, "$lte": 5})
    assert not _compare(5, {"$lt": 5})
    assert _compare("b", {"$ne": "a"})
    assert not _compare("x", {"$gt": 3})  # types that cannot be ordered never match
    assert not _compare(True, {"$gt": 0})
    assert _compare([1, 2], [1, 2]) and _compare({"a": 1}, {"a": 1})
    with pytest.raises(ValueError):
        _compare(1, {"$regex": "x"})
    assert _passes({"a": 1, "b": "x"}, {"a": 1, "b": {"$eq": "x"}})
    assert not _passes({"a": 1}, {"a": 2})
    assert _passes({"a": 1}, None)


def test_namespace_match_conditions():
    ns = ("a", "b", "c")
    assert _namespace_matches(ns, [MatchCondition("prefix", ("a",))])
    assert _namespace_matches(ns, [MatchCondition("prefix", ("a", "*"))])
    assert _namespace_matches(ns, [MatchCondition("suffix", ("c",))])
    assert _namespace_matches(ns, [MatchCondition("prefix", ("a",)), MatchCondition("suffix", ("b", "c"))])
    assert not _namespace_matches(ns, [MatchCondition("prefix", ("b",))])
    assert not _namespace_matches(("a",), [MatchCondition("prefix", ("a", "b"))])


def test_store_core_without_an_index_has_no_embeddings():
    core = StoreCore("p", None, None)
    assert core.embeddings is None and core.dims is None


def test_store_core_reads_the_index_config():
    core = StoreCore("p", {"dims": 3, "embed": lambda texts: [[0.0, 0.0, 0.0] for _ in texts], "fields": ["a"]}, None)
    assert core.dims == 3 and core.fields == ["a"] and core.embeddings is not None
