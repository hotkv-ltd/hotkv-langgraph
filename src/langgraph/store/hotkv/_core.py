# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""Storage logic of the HotKV store for LangGraph.

Like the checkpointer, it is written once as generators. A generator yields
a list of raw commands (`Tx` for atomic, `Soft` when an error reply is
acceptable), or an `Embed` request for texts to turn into vectors, and receives
the replies; its `return` value is the result.

Data model, with `p` the key prefix, `ns` a namespace joined with U+001E:

    p:it:ns SEP key   hash   v (JSON value), c (created), u (updated),
                             vk (JSON list of the item's vector keys), ttl (minutes)
    p:nk:ns           set    keys of a namespace
    p:ns              set    every namespace
    p:vi:ns           vector index of a namespace (VSIM.*), one vector per indexed text

Search without a query reads the items of the matching namespaces, so it is
meant for namespaces of up to a few thousand items (a user's memories). Search
with a query asks each matching namespace's vector index for the nearest texts
and filters after that.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

from langgraph.store.base import (
    GetOp,
    IndexConfig,
    InvalidNamespaceError,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
)

SEP = "\x1f"
NSEP = "\x1e"
FETCH_CHUNK = 200
MIN_NEIGHBOURS = 25


class Tx(list):
    """Commands that must run atomically."""


class Soft(list):
    """Commands whose error replies are returned in place instead of raised."""


class SoftTx(Tx, Soft):
    """Atomic commands where an error reply is returned in place (for CREATEINDEX on an index that exists)."""


class Embed(list):
    """Texts to embed; the reply is one vector per text."""

    def __init__(self, texts: Sequence[str], query: bool = False) -> None:
        super().__init__(texts)
        self.query = query


Flow = Generator[Any, Any, Any]


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


def _floats(vector: Sequence[float]) -> List[str]:
    return [format(float(x), ".9g") for x in vector]


def _validate(namespace: Tuple[str, ...], key: Optional[str] = None) -> None:
    if not namespace:
        raise InvalidNamespaceError("Namespace cannot be empty.")
    for label in namespace:
        if not isinstance(label, str):
            raise InvalidNamespaceError(f"Namespace labels must be strings, got {type(label).__name__}.")
        if "." in label:
            raise InvalidNamespaceError(f"Namespace labels cannot contain periods ('.'): {label!r}")
        if not label:
            raise InvalidNamespaceError("Namespace labels cannot be empty strings.")
        if NSEP in label or SEP in label:
            raise InvalidNamespaceError("Namespace labels cannot contain control characters U+001E or U+001F.")
    if namespace[0] == "langgraph":
        raise InvalidNamespaceError("Root label for namespace cannot be 'langgraph'.")
    if key is not None and (NSEP in key or SEP in key):
        raise ValueError("Keys cannot contain control characters U+001E or U+001F.")


def _ns_join(namespace: Sequence[str]) -> str:
    return NSEP.join(namespace)


def _ns_split(joined: str) -> Tuple[str, ...]:
    return tuple(joined.split(NSEP))


def _when(raw: Any) -> datetime:
    return datetime.fromtimestamp(float(_text(raw)), tz=timezone.utc)


def _compare(actual: Any, condition: Any) -> bool:
    if isinstance(condition, dict) and condition and all(k.startswith("$") for k in condition):
        for op in condition:
            if op not in _OPERATORS:
                raise ValueError(f"Unsupported filter operator: {op}")
        for op, want in condition.items():
            if op == "$eq" and not actual == want:
                return False
            if op == "$ne" and not actual != want:
                return False
            if op in ("$gt", "$gte", "$lt", "$lte"):
                if isinstance(actual, bool) or isinstance(want, bool):
                    return False
                if not (isinstance(actual, (int, float)) and isinstance(want, (int, float))) and not (
                    isinstance(actual, str) and isinstance(want, str)
                ):
                    return False
                if op == "$gt" and not actual > want:
                    return False
                if op == "$gte" and not actual >= want:
                    return False
                if op == "$lt" and not actual < want:
                    return False
                if op == "$lte" and not actual <= want:
                    return False
            if op not in ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte"):
                raise ValueError(f"Unsupported filter operator: {op}")
        return True
    return bool(actual == condition)


def _passes(value: Dict[str, Any], flt: Optional[Dict[str, Any]]) -> bool:
    if not flt:
        return True
    return all(_compare(value.get(k), cond) for k, cond in flt.items())


def _namespace_matches(ns: Tuple[str, ...], conditions: Sequence[Any]) -> bool:
    for cond in conditions:
        path = tuple(cond.path)
        if cond.match_type == "prefix":
            part = ns[: len(path)]
        elif cond.match_type == "suffix":
            part = ns[-len(path):] if path else ()
        else:
            raise ValueError(f"Unsupported match type: {cond.match_type}")
        if len(part) != len(path) or not all(p == "*" or p == n for p, n in zip(path, part)):
            return False
    return True


# Pushes an item's expiry out, but only if its ttl is still the one the caller read: a concurrent put that
# changed or removed the ttl must not be undone by a read.
_REFRESH_SCRIPT = (
    "if redis.call('HGET', KEYS[1], 'ttl') == ARGV[1] then return redis.call('EXPIRE', KEYS[1], ARGV[2]) end return 0"
)

# Forgets a key whose item expired, unless the item was written again since the caller looked.
_DROP_SCRIPT = (
    "if redis.call('EXISTS', KEYS[2]) == 0 then redis.call('SREM', KEYS[1], ARGV[1]); return 1 end return 0"
)

_OPERATORS = ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte")

_FORGET_SCRIPT = (
    "if redis.call('SCARD', KEYS[1]) == 0 then "
    "redis.call('SREM', KEYS[2], ARGV[1]); redis.pcall('VSIM.DROPINDEX', KEYS[3]); return 1 end return 0"
)


class StoreCore:
    def __init__(self, prefix: str, index: Optional[IndexConfig], ttl: Optional[TTLConfig]) -> None:
        self.p = prefix
        self.index = index
        self.ttl = ttl or {}
        self.dims: Optional[int] = None
        self.fields: List[str] = ["$"]
        self.embeddings: Any = None
        if index:
            self.dims = int(index["dims"])
            self.embeddings = ensure_embeddings(index.get("embed"))
            self.fields = list(index.get("fields") or ["$"])

    # -- keys -------------------------------------------------------------------

    def item_key(self, ns: str, key: str) -> str:
        return f"{self.p}:it:{ns}{SEP}{key}"

    def keys_key(self, ns: str) -> str:
        return f"{self.p}:nk:{ns}"

    @property
    def namespaces_key(self) -> str:
        return f"{self.p}:ns"

    def vector_index(self, ns: str) -> str:
        return f"{self.p}:vi:{ns}"

    # -- batch ------------------------------------------------------------------

    def batch(self, ops: Sequence[Op]) -> Flow:
        results: List[Any] = [None] * len(ops)
        i = 0
        while i < len(ops):
            op = ops[i]
            if isinstance(op, GetOp):
                j = i
                while j < len(ops) and isinstance(ops[j], GetOp):
                    j += 1
                results[i:j] = yield from self._get_many(list(ops[i:j]))  # type: ignore[arg-type]
                i = j
                continue
            if isinstance(op, PutOp):
                yield from self._put(op)
            elif isinstance(op, SearchOp):
                results[i] = yield from self._search(op)
            elif isinstance(op, ListNamespacesOp):
                results[i] = yield from self._list_namespaces(op)
            else:
                raise ValueError(f"Unknown operation type: {type(op).__name__}")
            i += 1
        return results

    # -- get --------------------------------------------------------------------

    def _item(self, ns: str, key: str, head: Dict[bytes, bytes], score: Optional[float] = None) -> Item:
        common = dict(
            namespace=_ns_split(ns), key=key, value=json.loads(head[b"v"]),
            created_at=_when(head[b"c"]), updated_at=_when(head[b"u"]),
        )
        if score is None:
            return Item(**common)
        return SearchItem(score=score, **common)

    def _get_many(self, ops: List[GetOp]) -> Flow:
        for op in ops:
            _validate(op.namespace, op.key)
        joined = [_ns_join(op.namespace) for op in ops]
        replies = yield [("HGETALL", self.item_key(ns, op.key)) for ns, op in zip(joined, ops)]
        out: List[Optional[Item]] = []
        touch: List[Tuple] = []
        for ns, op, head in zip(joined, ops, replies):
            if not head:
                out.append(None)
                continue
            out.append(self._item(ns, op.key, head))
            self._refresh(touch, ns, op.key, head, op.refresh_ttl)
        if touch:
            yield Soft(touch)
        return out

    def _refresh(self, cmds: List[Tuple], ns: str, key: str, head: Dict[bytes, bytes], wanted: bool) -> None:
        minutes = _text(head.get(b"ttl", b""))
        if wanted and self.ttl.get("refresh_on_read", True) and minutes:
            cmds.append((
                "EVAL", _REFRESH_SCRIPT, 1, self.item_key(ns, key), minutes, math.ceil(float(minutes) * 60),
            ))

    # -- put / delete -----------------------------------------------------------

    def _put(self, op: PutOp) -> Flow:
        _validate(op.namespace, op.key)
        ns = _ns_join(op.namespace)
        if op.value is None:
            yield from self._delete(ns, op.key)
            return
        item = self.item_key(ns, op.key)
        ((old_created, old_vectors),) = yield [("HMGET", item, "c", "vk")]
        stale_vectors: List[str] = json.loads(_text(old_vectors)) if old_vectors else []
        now = time.time()
        minutes = op.ttl if op.ttl is not None else self.ttl.get("default_ttl")

        paths: List[str] = []
        if self.embeddings is not None and op.index is not False:
            paths = list(op.index) if isinstance(op.index, list) else self.fields
        texts: List[str] = []
        vector_keys: List[str] = []
        for path in paths:
            for n, text in enumerate(get_text_at_path(op.value, path)):
                texts.append(text)
                vector_keys.append(f"{op.key}{SEP}{path}{SEP}{n}")
        vectors: List[Any] = []
        if texts:
            vectors = yield Embed(texts)
            for vec in vectors:
                if len(vec) != self.dims:
                    raise ValueError(f"embedding has {len(vec)} dimensions, the store index is configured for {self.dims}")

        index = self.vector_index(ns)
        cmds: List[Tuple] = [
            (
                "HSET", item,
                "v", json.dumps(op.value), "c", _text(old_created) if old_created else repr(now), "u", repr(now),
                "vk", json.dumps(vector_keys), "ttl", "" if minutes is None else repr(float(minutes)),
            ),
            ("EXPIRE", item, math.ceil(float(minutes) * 60)) if minutes is not None else ("PERSIST", item),
            # the key set goes first: while the namespace is not empty, a concurrent delete will not drop its index
            ("SADD", self.keys_key(ns), op.key),
            ("SADD", self.namespaces_key, ns),
        ]
        if vector_keys:
            # created here, in the same transaction, so a writer never meets an index another process dropped
            cmds.append(("VSIM.CREATEINDEX", index, self.dims, "METRIC", "cosine"))
            for vkey, vec in zip(vector_keys, vectors):
                cmds.append(("VSIM.SET", index, vkey, *_floats(vec)))
        replies = yield (SoftTx(cmds) if vector_keys else Tx(cmds))
        for cmd, reply in zip(cmds, replies):
            if isinstance(reply, Exception) and not (cmd[0] == "VSIM.CREATEINDEX" and "already exists" in str(reply)):
                raise reply
        # Only now remove what the new version no longer has: the item stays searchable throughout, and a
        # failed write above leaves the old vectors in place.
        dropped = [v for v in stale_vectors if v not in set(vector_keys)]
        if dropped:
            yield Soft([("VSIM.DEL", index, v) for v in dropped])

    def _delete(self, ns: str, key: str) -> Flow:
        item = self.item_key(ns, key)
        (old_vectors,) = yield [("HGET", item, "vk")]
        index = self.vector_index(ns)
        if old_vectors:
            yield Soft([("VSIM.DEL", index, v) for v in json.loads(_text(old_vectors))])
        yield Tx([("UNLINK", item), ("SREM", self.keys_key(ns), key)])
        yield from self._forget_if_empty(ns)

    def _forget_if_empty(self, ns: str) -> Flow:
        # one atomic script: a writer that added an item since we looked keeps its namespace and index
        yield Soft([("EVAL", _FORGET_SCRIPT, 3, self.keys_key(ns), self.namespaces_key, self.vector_index(ns), ns)])

    # -- search -----------------------------------------------------------------

    def _matching_namespaces(self, prefix: Tuple[str, ...]) -> Flow:
        (members,) = yield [("SMEMBERS", self.namespaces_key)]
        found = [_text(m) for m in members]
        return sorted(ns for ns in found if _ns_split(ns)[: len(prefix)] == tuple(prefix))

    def _fetch(self, refs: List[Tuple[str, str]]) -> Flow:
        """HGETALL replies for (ns, key) pairs, in chunks."""
        heads: List[Dict[bytes, bytes]] = []
        for start in range(0, len(refs), FETCH_CHUNK):
            part = refs[start:start + FETCH_CHUNK]
            heads.extend((yield [("HGETALL", self.item_key(ns, key)) for ns, key in part]))
        return heads

    def _search(self, op: SearchOp) -> Flow:
        matching = yield from self._matching_namespaces(op.namespace_prefix)
        if not matching:
            return []
        if op.query and self.embeddings is not None:
            return (yield from self._semantic(op, matching))
        replies = yield [("SMEMBERS", self.keys_key(ns)) for ns in matching]
        refs = [(ns, _text(k)) for ns, keys in zip(matching, replies) for k in keys]
        heads = yield from self._fetch(refs)
        rows: List[Tuple[Item, Dict[bytes, bytes]]] = []
        gone: List[Tuple[str, str]] = []
        for (ns, key), head in zip(refs, heads):
            if not head:
                gone.append((ns, key))
                continue
            item = self._item(ns, key, head)
            if _passes(item.value, op.filter):
                rows.append((item, head))
        yield from self._drop_missing(gone)
        rows.sort(key=lambda r: (r[0].updated_at, r[0].key), reverse=True)
        page = rows[op.offset: op.offset + op.limit]
        touch: List[Tuple] = []
        out: List[SearchItem] = []
        for item, head in page:
            ns = _ns_join(item.namespace)
            self._refresh(touch, ns, item.key, head, op.refresh_ttl)
            out.append(SearchItem(item.namespace, item.key, item.value, item.created_at, item.updated_at, None))
        if touch:
            yield Soft(touch)
        return out

    def _semantic(self, op: SearchOp, matching: List[str]) -> Flow:
        (query_vector,) = yield Embed([op.query], query=True)
        wanted = max(MIN_NEIGHBOURS, (op.offset + op.limit) * 10)
        replies = yield Soft([
            ("VSIM.SEARCH", self.vector_index(ns), wanted, *_floats(query_vector)) for ns in matching
        ])
        candidates: List[Tuple[str, str, str, float]] = []  # (namespace, item key, vector key, similarity)
        for ns, reply in zip(matching, replies):
            if isinstance(reply, Exception):
                continue  # a namespace nothing was indexed in
            for vkey, distance in reply:
                text = _text(vkey)
                candidates.append((ns, text.split(SEP, 1)[0], text, 1.0 - float(distance)))
        refs = sorted({(ns, key) for ns, key, _, _ in candidates})
        heads = yield from self._fetch(refs)
        head_of = dict(zip(refs, heads))
        owned: Dict[Tuple[str, str], set] = {}
        best: Dict[Tuple[str, str], float] = {}
        stale: List[Tuple[str, str]] = []
        for ns, key, vkey, score in candidates:
            head = head_of[(ns, key)]
            if not head:
                stale.append((ns, vkey))  # the item expired or was deleted
                continue
            if (ns, key) not in owned:
                owned[(ns, key)] = set(json.loads(_text(head.get(b"vk", b"[]")) or "[]"))
            if vkey not in owned[(ns, key)]:
                stale.append((ns, vkey))  # a vector of an older version of the item
                continue
            if score > best.get((ns, key), -math.inf):
                best[(ns, key)] = score
        rows: List[Tuple[SearchItem, Dict[bytes, bytes]]] = []
        for (ns, key), score in best.items():
            item = self._item(ns, key, head_of[(ns, key)], score=score)
            if _passes(item.value, op.filter):
                rows.append((item, head_of[(ns, key)]))  # type: ignore[arg-type]
        if stale:
            yield Soft([("VSIM.DEL", self.vector_index(ns), v) for ns, v in stale])
        gone = [ref for ref in refs if not head_of[ref]]
        if gone:
            yield from self._drop_missing(gone)
        rows.sort(key=lambda r: r[0].score or 0.0, reverse=True)
        page = rows[op.offset: op.offset + op.limit]
        touch: List[Tuple] = []
        for item, head in page:
            self._refresh(touch, _ns_join(item.namespace), item.key, head, op.refresh_ttl)
        if touch:
            yield Soft(touch)
        return [item for item, _ in page]

    def _drop_missing(self, gone: List[Tuple[str, str]]) -> Flow:
        """Forget keys whose item expired."""
        if not gone:
            return
        yield Soft([
            ("EVAL", _DROP_SCRIPT, 2, self.keys_key(ns), self.item_key(ns, key), key) for ns, key in gone
        ])
        for ns in sorted({ns for ns, _ in gone}):
            yield from self._forget_if_empty(ns)

    # -- namespaces -------------------------------------------------------------

    def _list_namespaces(self, op: ListNamespacesOp) -> Flow:
        (members,) = yield [("SMEMBERS", self.namespaces_key)]
        found = [_ns_split(_text(m)) for m in members]
        if op.match_conditions:
            found = [ns for ns in found if _namespace_matches(ns, op.match_conditions)]
        if op.max_depth is not None:
            found = [ns[: op.max_depth] for ns in found]
        unique = sorted(set(found))
        return unique[op.offset: op.offset + op.limit]
