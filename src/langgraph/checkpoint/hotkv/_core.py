# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Storage logic of the HotKV checkpointer.

Everything is written once, as generators. A generator yields a list of raw
commands (a `Tx` when they must run atomically) and receives the list of
replies; its `return` value is the result. `HotKVSaver` drives these
generators with the synchronous client and `AsyncHotKVSaver` with the asyncio
client, so the two cannot drift apart.

Data model, with `p` the key prefix, `t` the thread id and `ns` the checkpoint
namespace:

    p:cp:t SEP ns SEP id     hash   cp (checkpoint without channel values), md
                                    (metadata), parent (id or empty), run (run id)
    p:bl:t SEP ns SEP ch SEP v      string  one channel value at one channel version
    p:wr:t SEP ns SEP id     hash   pending writes: field `task SEP idx`
    p:ix:t SEP ns            zset   checkpoint ids, all score 0, so lexicographic order
                                    is time order (ids are time-ordered uuid6)
    p:ns:t                   set    namespaces of a thread
    p:bk:t SEP ns            set    `ch SEP v` of every stored blob, for cleanup
    p:th                     set    every thread id
    p:run:run_id             set    `t SEP ns SEP id` of the checkpoints of a run
    p:gn:t SEP ns            string counts the writes to a thread and namespace; cleanup only removes
                                    channel values if nothing was written since it looked

A channel value is stored once per channel version, so a step that changes one
channel writes one value, not the whole state.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import _DeltaSnapshot

SEP = "\x1f"
EMPTY = b"\xffempty"
CHUNK = 100


class Tx(list):
    """A batch of commands that must run atomically (MULTI/EXEC)."""


Step = Any
Flow = Generator[Step, List[Any], Any]


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


def _pack(*parts: bytes) -> bytes:
    return b"".join(struct.pack(">I", len(p)) + p for p in parts)


def _unpack(raw: bytes) -> List[bytes]:
    parts, pos = [], 0
    while pos < len(raw):
        (size,) = struct.unpack_from(">I", raw, pos)
        pos += 4
        parts.append(raw[pos:pos + size])
        pos += size
    return parts


def _check(name: str, value: str) -> str:
    if SEP in value:
        raise ValueError(f"{name} must not contain the control character U+001F")
    return value


class Layout:
    """Key names."""

    def __init__(self, prefix: str) -> None:
        self.p = prefix

    def cp(self, t: str, ns: str, cid: str) -> str:
        return f"{self.p}:cp:{t}{SEP}{ns}{SEP}{cid}"

    def wr(self, t: str, ns: str, cid: str) -> str:
        return f"{self.p}:wr:{t}{SEP}{ns}{SEP}{cid}"

    def blob(self, t: str, ns: str, ch: str, ver: Any) -> str:
        return f"{self.p}:bl:{t}{SEP}{ns}{SEP}{ch}{SEP}{ver}"

    def blob_of_ref(self, t: str, ns: str, ref: str) -> str:
        return f"{self.p}:bl:{t}{SEP}{ns}{SEP}{ref}"

    def ix(self, t: str, ns: str) -> str:
        return f"{self.p}:ix:{t}{SEP}{ns}"

    def nsset(self, t: str) -> str:
        return f"{self.p}:ns:{t}"

    def gen(self, t: str, ns: str) -> str:
        return f"{self.p}:gn:{t}{SEP}{ns}"

    def bset(self, t: str, ns: str) -> str:
        return f"{self.p}:bk:{t}{SEP}{ns}"

    @property
    def threads(self) -> str:
        return f"{self.p}:th"

    def run(self, run_id: str) -> str:
        return f"{self.p}:run:{run_id}"


def _ref(ch: str, ver: Any) -> str:
    return f"{ch}{SEP}{ver}"


def _config(t: str, ns: str, cid: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": t, "checkpoint_ns": ns, "checkpoint_id": cid}}


# Removes channel values that no checkpoint uses, but only if nothing was written to the thread since the
# caller decided which they are (ARGV[1] is the write counter it saw, ARGV[2] the key prefix of the values).
_UNLINK_STALE_BLOBS = (
    "local g = redis.call('GET', KEYS[1]); if (g or '') ~= ARGV[1] then return 0 end; "
    "for i = 3, #ARGV do redis.call('UNLINK', ARGV[2] .. ARGV[i]); redis.call('SREM', KEYS[2], ARGV[i]) end; return 1"
)

# Forgets a namespace (and the thread, if that was its last one) when it has no checkpoints, atomically, so a
# checkpoint written in the meantime keeps its namespace and thread listed.
_RETIRE_NAMESPACE = (
    "if redis.call('ZCARD', KEYS[1]) ~= 0 then return 0 end; "
    "redis.call('SREM', KEYS[2], ARGV[1]); "
    "if redis.call('SCARD', KEYS[4]) == 0 then redis.call('UNLINK', KEYS[4]) end; "
    "redis.call('UNLINK', KEYS[5]); "
    "if redis.call('SCARD', KEYS[2]) == 0 then redis.call('SREM', KEYS[3], ARGV[2]) end; return 1"
)


class Core:
    def __init__(self, serde: SerializerProtocol, prefix: str) -> None:
        self.serde = serde
        self.L = Layout(prefix)

    # -- values ---------------------------------------------------------------

    def dump(self, obj: Any) -> bytes:
        kind, data = self.serde.dumps_typed(obj)
        return kind.encode("ascii") + b"\x00" + data

    def load(self, raw: bytes) -> Any:
        kind, _, data = bytes(raw).partition(b"\x00")
        return self.serde.loads_typed((kind.decode("ascii"), data))

    # -- put / put_writes -------------------------------------------------------

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Flow:
        conf = config["configurable"]
        t = _check("thread_id", conf["thread_id"])
        ns = _check("checkpoint_ns", conf.get("checkpoint_ns", ""))
        parent = conf.get("checkpoint_id")
        stored = dict(checkpoint)
        values = stored.pop("channel_values")
        md = get_checkpoint_metadata(config, metadata)
        run_id = str(md.get("run_id") or "")
        cid = checkpoint["id"]
        L = self.L
        cmds: List[Tuple] = []
        for ch, ver in new_versions.items():
            cmds.append(("SET", L.blob(t, ns, ch, ver), self.dump(values[ch]) if ch in values else EMPTY))
            cmds.append(("SADD", L.bset(t, ns), _ref(ch, ver)))
        cmds.append((
            "HSET", L.cp(t, ns, cid),
            "cp", self.dump(stored), "md", self.dump(md), "parent", parent or "", "run", run_id,
        ))
        cmds.append(("ZADD", L.ix(t, ns), 0, cid))
        cmds.append(("SADD", L.nsset(t), ns))
        cmds.append(("SADD", L.threads, t))
        cmds.append(("INCR", L.gen(t, ns)))
        if run_id:
            cmds.append(("SADD", L.run(run_id), f"{t}{SEP}{ns}{SEP}{cid}"))
        yield Tx(cmds)
        return _config(t, ns, cid)

    def put_writes(
        self, config: Dict[str, Any], writes: Sequence[Tuple[str, Any]], task_id: str, task_path: str = ""
    ) -> Flow:
        conf = config["configurable"]
        t = _check("thread_id", conf["thread_id"])
        ns = _check("checkpoint_ns", conf.get("checkpoint_ns", ""))
        cid = conf["checkpoint_id"]
        key = self.L.wr(t, ns, cid)
        cmds: List[Tuple] = []
        for idx, (channel, value) in enumerate(writes):
            special = channel in WRITES_IDX_MAP
            widx = WRITES_IDX_MAP[channel] if special else idx
            packed = _pack(channel.encode("utf-8"), task_path.encode("utf-8"), self.dump(value))
            # a special write (error, interrupt, resume) replaces the earlier one; others are written once
            cmds.append(("HSET" if special else "HSETNX", key, f"{task_id}{SEP}{widx}", packed))
        if cmds:
            yield Tx(cmds)

    # -- reads ------------------------------------------------------------------

    def _tuples(self, refs: List[Tuple[str, str, str]]) -> Flow:
        """Checkpoint tuples for (thread, ns, id) references, None where one is gone."""
        if not refs:
            return []
        L = self.L
        cmds: List[Tuple] = []
        for t, ns, cid in refs:
            cmds.append(("HGETALL", L.cp(t, ns, cid)))
            cmds.append(("HGETALL", L.wr(t, ns, cid)))
        replies = yield cmds
        heads: List[Optional[Tuple[Dict[bytes, bytes], Dict[bytes, bytes], Dict[str, Any]]]] = []
        blob_keys: List[str] = []
        for i, (t, ns, cid) in enumerate(refs):
            head, writes = replies[2 * i], replies[2 * i + 1]
            if not head:
                heads.append(None)
                continue
            checkpoint = self.load(head[b"cp"])
            heads.append((head, writes, checkpoint))
            blob_keys.extend(L.blob(t, ns, ch, ver) for ch, ver in checkpoint["channel_versions"].items())
        blobs: List[Any] = []
        if blob_keys:
            blobs = (yield [("MGET", *blob_keys)])[0]
        out: List[Optional[CheckpointTuple]] = []
        pos = 0
        for (t, ns, cid), item in zip(refs, heads):
            if item is None:
                out.append(None)
                continue
            head, writes, checkpoint = item
            values: Dict[str, Any] = {}
            for ch in checkpoint["channel_versions"]:
                raw = blobs[pos]
                pos += 1
                if raw is not None and raw != EMPTY:
                    values[ch] = self.load(raw)
            checkpoint["channel_values"] = values
            pending: List[Tuple[str, str, Any]] = []
            for field in sorted(writes, key=_write_order):
                channel, _path, packed = _unpack(writes[field])
                task_id = _text(field).rsplit(SEP, 1)[0]
                pending.append((task_id, channel.decode("utf-8"), self.load(packed)))
            parent = _text(head.get(b"parent", b""))
            out.append(CheckpointTuple(
                config=_config(t, ns, cid),
                checkpoint=checkpoint,
                metadata=self.load(head[b"md"]),
                parent_config=_config(t, ns, parent) if parent else None,
                pending_writes=pending,
            ))
        return out

    def get_tuple(self, config: Dict[str, Any]) -> Flow:
        conf = config["configurable"]
        t = conf["thread_id"]
        ns = conf.get("checkpoint_ns", "")
        cid = get_checkpoint_id(config)
        if not cid:
            (ids,) = yield [("ZREVRANGEBYLEX", self.L.ix(t, ns), "+", "-", "LIMIT", 0, 1)]
            if not ids:
                return None
            cid = _text(ids[0])
        found = yield from self._tuples([(t, ns, cid)])
        return found[0]

    def _all_threads(self) -> Flow:
        threads: List[str] = []
        cursor: Any = 0
        while True:
            ((cursor, items),) = yield [("SSCAN", self.L.threads, cursor, "COUNT", 1000)]
            threads.extend(_text(i) for i in items)
            if int(cursor) == 0:
                return threads

    def list(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Flow:
        conf = config["configurable"] if config else {}
        thread = conf.get("thread_id")
        ns_filter = conf.get("checkpoint_ns")
        id_filter = conf.get("checkpoint_id")
        before_id = get_checkpoint_id(before) if before else None
        threads = [thread] if thread is not None else (yield from self._all_threads())
        ns_by_thread: Dict[str, List[str]]
        if ns_filter is not None:
            ns_by_thread = {t: [ns_filter] for t in threads}
        else:
            replies = yield [("SMEMBERS", self.L.nsset(t)) for t in threads]
            ns_by_thread = {t: sorted(_text(n) for n in r) for t, r in zip(threads, replies)}
        groups = [(t, ns) for t in threads for ns in ns_by_thread[t]]
        upper = f"({before_id}" if before_id else "+"
        replies = yield [("ZREVRANGEBYLEX", self.L.ix(t, ns), upper, "-") for t, ns in groups]
        candidates = [(t, ns, _text(cid)) for (t, ns), ids in zip(groups, replies) for cid in ids]
        if id_filter:
            candidates = [c for c in candidates if c[2] == id_filter]
        candidates.sort(key=lambda c: c[2], reverse=True)
        out: List[CheckpointTuple] = []
        pos = 0
        step = limit if (limit and not filter) else CHUNK
        while pos < len(candidates) and (limit is None or len(out) < limit):
            chunk = candidates[pos:pos + step]
            pos += step
            if filter:
                mds = yield [("HGET", self.L.cp(*c), "md") for c in chunk]
                chunk = [
                    c for c, raw in zip(chunk, mds)
                    if raw is not None and _matches(self.load(raw), filter)
                ]
            if limit is not None:
                chunk = chunk[: limit - len(out)]
            out.extend(tup for tup in (yield from self._tuples(chunk)) if tup is not None)
        return out

    # -- deletes ----------------------------------------------------------------

    def _delete_checkpoints(self, t: str, ns: str, ids: List[str], *, everything: bool = False) -> Flow:
        """Remove checkpoints (with their writes and run entries), then the channel values nothing refers to."""
        L = self.L
        if ids:
            runs = yield [("HGET", L.cp(t, ns, i), "run") for i in ids]
            cmds: List[Tuple] = []
            for i, run in zip(ids, runs):
                if run:
                    cmds.append(("SREM", L.run(_text(run)), f"{t}{SEP}{ns}{SEP}{i}"))
                cmds.append(("UNLINK", L.cp(t, ns, i), L.wr(t, ns, i)))
            cmds.append(("ZREM", L.ix(t, ns), *ids))
            yield Tx(cmds)
        yield from self._collect_blobs(t, ns, everything)
        yield [("EVAL", _RETIRE_NAMESPACE, 5, L.ix(t, ns), L.nsset(t), L.threads, L.bset(t, ns), L.gen(t, ns), ns, t)]

    def _collect_blobs(self, t: str, ns: str, everything: bool) -> Flow:
        """Delete the channel values no checkpoint refers to.

        The values are listed first and the checkpoints read afterwards, and the delete only goes ahead if no
        checkpoint was written to the thread in between (a new checkpoint may reuse an existing value). When one
        was, the pass is repeated; a value that is still left is collected by the next cleanup.
        """
        L = self.L
        prefix = f"{L.p}:bl:{t}{SEP}{ns}{SEP}"
        for _ in range(3):
            (gen,) = yield [("GET", L.gen(t, ns))]
            (members,) = yield [("SMEMBERS", L.bset(t, ns))]
            referenced: set = set()
            if not everything:
                (ids,) = yield [("ZRANGEBYLEX", L.ix(t, ns), "-", "+")]
                if ids:
                    raws = yield [("HGET", L.cp(t, ns, _text(i)), "cp") for i in ids]
                    for raw in raws:
                        if raw:
                            for ch, ver in self.load(raw)["channel_versions"].items():
                                referenced.add(_ref(ch, ver))
            stale = [_text(m) for m in members if _text(m) not in referenced]
            if not stale:
                return
            (done,) = yield [("EVAL", _UNLINK_STALE_BLOBS, 2, L.gen(t, ns), L.bset(t, ns), "" if gen is None else _text(gen), prefix, *stale)]
            if done:
                return

    def _ids(self, t: str, ns: str) -> Flow:
        (ids,) = yield [("ZRANGEBYLEX", self.L.ix(t, ns), "-", "+")]
        return [_text(i) for i in ids]

    def _namespaces(self, t: str) -> Flow:
        (nss,) = yield [("SMEMBERS", self.L.nsset(t))]
        return sorted(_text(n) for n in nss)

    def delete_thread(self, thread_id: str) -> Flow:
        for ns in (yield from self._namespaces(thread_id)):
            ids = yield from self._ids(thread_id, ns)
            yield from self._delete_checkpoints(thread_id, ns, ids, everything=True)

    def delete_for_runs(self, run_ids: Sequence[str]) -> Flow:
        if not run_ids:
            return
        sets = yield [("SMEMBERS", self.L.run(r)) for r in run_ids]
        by_group: Dict[Tuple[str, str], List[str]] = {}
        for members in sets:
            for member in members:
                t, ns, cid = _text(member).split(SEP, 2)
                by_group.setdefault((t, ns), []).append(cid)
        for (t, ns), ids in by_group.items():
            yield from self._delete_checkpoints(t, ns, ids)
        yield [("UNLINK", self.L.run(r)) for r in run_ids]

    def copy_thread(self, source: str, target: str) -> Flow:
        L = self.L
        cmds: List[Tuple] = []
        for ns in (yield from self._namespaces(source)):
            ids = yield from self._ids(source, ns)
            if not ids:
                continue
            reads: List[Tuple] = []
            for i in ids:
                reads.append(("HGETALL", L.cp(source, ns, i)))
                reads.append(("HGETALL", L.wr(source, ns, i)))
            reads.append(("SMEMBERS", L.bset(source, ns)))
            replies = yield reads
            refs = [_text(m) for m in replies[-1]]
            blobs: List[Any] = []
            if refs:
                blobs = (yield [("MGET", *[L.blob_of_ref(source, ns, r) for r in refs])])[0]
            for r, raw in zip(refs, blobs):
                if raw is not None:
                    cmds.append(("SET", L.blob_of_ref(target, ns, r), raw))
                    cmds.append(("SADD", L.bset(target, ns), r))
            for n, cid in enumerate(ids):
                head, writes = replies[2 * n], replies[2 * n + 1]
                if not head:
                    continue
                flat: List[Any] = []
                for k, v in head.items():
                    flat += [k, v]
                cmds.append(("HSET", L.cp(target, ns, cid), *flat))
                if writes:
                    flat = []
                    for k, v in writes.items():
                        flat += [k, v]
                    cmds.append(("HSET", L.wr(target, ns, cid), *flat))
                cmds.append(("ZADD", L.ix(target, ns), 0, cid))
                run = head.get(b"run")
                if run:
                    cmds.append(("SADD", L.run(_text(run)), f"{target}{SEP}{ns}{SEP}{cid}"))
            cmds.append(("SADD", L.nsset(target), ns))
        if cmds:
            cmds.append(("SADD", L.threads, target))
            yield Tx(cmds)

    def prune(self, thread_ids: Sequence[str], strategy: str = "keep_latest") -> Flow:
        if strategy not in ("keep_latest", "delete"):
            raise ValueError(f"unknown prune strategy: {strategy!r}")
        for t in thread_ids:
            for ns in (yield from self._namespaces(t)):
                ids = yield from self._ids(t, ns)
                if not ids:
                    continue
                if strategy == "delete":
                    yield from self._delete_checkpoints(t, ns, ids, everything=True)
                    continue
                keep = yield from self._keep_with_delta_history(t, ns, ids[-1])
                yield from self._delete_checkpoints(t, ns, [i for i in ids if i not in keep])

    def _keep_with_delta_history(self, t: str, ns: str, latest: str) -> Flow:
        """The checkpoints to keep so a DeltaChannel can still be rebuilt.

        A DeltaChannel stores only changes between snapshots. If the latest
        checkpoint says some channels count changes since a snapshot, its
        ancestors are kept back to one that holds a snapshot of each of them.
        """
        L = self.L
        keep = [latest]
        (raw,) = yield [("HGET", L.cp(t, ns, latest), "md")]
        counters = self.load(raw).get("counters_since_delta_snapshot") if raw else None
        need = set(counters or {})
        cursor = latest
        while need:
            head = (yield [("HGETALL", L.cp(t, ns, cursor))])[0]
            if not head:
                break
            versions = self.load(head[b"cp"])["channel_versions"]
            wanted = [ch for ch in need if ch in versions]
            if wanted:
                vals = (yield [("MGET", *[L.blob(t, ns, ch, versions[ch]) for ch in wanted])])[0]
                for ch, blob in zip(wanted, vals):
                    if blob is not None and blob != EMPTY and isinstance(self.load(blob), _DeltaSnapshot):
                        need.discard(ch)
            parent = _text(head.get(b"parent", b""))
            if not need or not parent:
                break
            keep.append(parent)
            cursor = parent
        return set(keep)


def _write_order(field: bytes) -> Tuple[str, int]:
    task_id, _, idx = _text(field).rpartition(SEP)
    return task_id, int(idx)


def _matches(metadata: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    return all(metadata.get(k) == v for k, v in flt.items())
