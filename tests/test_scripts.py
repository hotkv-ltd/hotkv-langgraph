# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""The guard scripts, driven in the exact interleavings they exist for (no timing involved)."""

import hotkv
import pytest

from langgraph.checkpoint.hotkv._core import _RETIRE_NAMESPACE, _UNLINK_STALE_BLOBS, SEP

from conftest import URL


@pytest.fixture
def client(prefix):
    c = hotkv.from_url(URL)
    yield c
    c.close()


def keys_left(client, prefix):
    cursor, found = 0, []
    while True:
        cursor, keys = client.scan(cursor, match=f"{prefix}:*", count=500)
        found += [k.decode() for k in keys]
        if cursor == 0:
            return sorted(found)


def test_stale_blobs_are_kept_when_a_checkpoint_was_written_since_they_were_judged_stale(client, prefix):
    gen, bset, blob_prefix = f"{prefix}:gn:t{SEP}", f"{prefix}:bk:t{SEP}", f"{prefix}:bl:t{SEP}{SEP}"
    client.set(gen, "5")
    client.sadd(bset, f"v{SEP}1", f"v{SEP}2")
    client.set(blob_prefix + f"v{SEP}1", "one")
    client.set(blob_prefix + f"v{SEP}2", "two")
    # the cleaner read counter 4; a put has raised it to 5 since: nothing may be deleted
    assert client.execute_command("EVAL", _UNLINK_STALE_BLOBS, 2, gen, bset, "4", blob_prefix, f"v{SEP}1") == 0
    assert client.get(blob_prefix + f"v{SEP}1") == b"one" and client.sismember(bset, f"v{SEP}1")
    # the counter is unchanged: the stale one goes, the other stays
    assert client.execute_command("EVAL", _UNLINK_STALE_BLOBS, 2, gen, bset, "5", blob_prefix, f"v{SEP}1") == 1
    assert client.get(blob_prefix + f"v{SEP}1") is None and not client.sismember(bset, f"v{SEP}1")
    assert client.get(blob_prefix + f"v{SEP}2") == b"two"


def test_a_thread_that_never_counted_writes_can_still_be_cleaned(client, prefix):
    bset, blob_prefix = f"{prefix}:bk:t{SEP}", f"{prefix}:bl:t{SEP}{SEP}"
    client.sadd(bset, f"v{SEP}1")
    client.set(blob_prefix + f"v{SEP}1", "one")
    assert client.execute_command("EVAL", _UNLINK_STALE_BLOBS, 2, f"{prefix}:gn:t{SEP}", bset, "", blob_prefix, f"v{SEP}1") == 1
    assert keys_left(client, prefix) == []


def test_a_namespace_with_a_checkpoint_is_not_retired(client, prefix):
    ix, nsset, threads, bset, gen = (f"{prefix}:ix:t{SEP}", f"{prefix}:ns:t", f"{prefix}:th", f"{prefix}:bk:t{SEP}", f"{prefix}:gn:t{SEP}")
    client.zadd(ix, {"cp1": 0})
    client.sadd(nsset, "")
    client.sadd(threads, "t")
    client.set(gen, "1")
    # the cleaner saw the namespace empty, but a checkpoint arrived before the script ran
    assert client.execute_command("EVAL", _RETIRE_NAMESPACE, 5, ix, nsset, threads, bset, gen, "", "t") == 0
    assert client.sismember(nsset, "") and client.sismember(threads, "t") and client.get(gen) == b"1"


def test_an_empty_namespace_is_retired_and_the_thread_with_its_last_one(client, prefix):
    ix1, ix2 = f"{prefix}:ix:t{SEP}", f"{prefix}:ix:t{SEP}sub"
    nsset, threads, bset1, bset2 = f"{prefix}:ns:t", f"{prefix}:th", f"{prefix}:bk:t{SEP}", f"{prefix}:bk:t{SEP}sub"
    gen1, gen2 = f"{prefix}:gn:t{SEP}", f"{prefix}:gn:t{SEP}sub"
    client.sadd(nsset, "", "sub")
    client.sadd(threads, "t")
    client.set(gen1, "3")
    client.set(gen2, "2")
    assert client.execute_command("EVAL", _RETIRE_NAMESPACE, 5, ix1, nsset, threads, bset1, gen1, "", "t") == 1
    assert client.smembers(nsset) == {b"sub"} and client.sismember(threads, "t")  # another namespace is left
    assert client.get(gen1) is None
    assert client.execute_command("EVAL", _RETIRE_NAMESPACE, 5, ix2, nsset, threads, bset2, gen2, "sub", "t") == 1
    assert keys_left(client, prefix) == []


def test_blob_references_survive_when_the_namespace_is_retired(client, prefix):
    """Retiring must not forget values that cleanup could not delete yet (their reference set stays)."""
    ix, nsset, threads, bset, gen = (f"{prefix}:ix:t{SEP}", f"{prefix}:ns:t", f"{prefix}:th", f"{prefix}:bk:t{SEP}", f"{prefix}:gn:t{SEP}")
    client.sadd(nsset, "")
    client.sadd(threads, "t")
    client.sadd(bset, f"v{SEP}1")
    assert client.execute_command("EVAL", _RETIRE_NAMESPACE, 5, ix, nsset, threads, bset, gen, "", "t") == 1
    assert client.sismember(bset, f"v{SEP}1"), "the record of a leftover value was dropped, so it could never be collected"
