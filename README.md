# LangGraph Checkpointer and Store for HotKV

[![CI](https://github.com/hotkv-ltd/hotkv-langgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/hotkv-ltd/hotkv-langgraph/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Persistence for [LangGraph](https://langchain-ai.github.io/langgraph/) agents on
[HotKV](https://hotkv.com), a RESP2/RESP3-compatible in-memory data store with built-in AI and
LLM-serving features (semantic and prompt caching, agent memory, RAG, feature store, rate limiting,
time series and more).

- **`HotKVSaver` / `AsyncHotKVSaver`**: checkpointers. A graph keeps its state between runs, can be
  interrupted and resumed, and can go back to any earlier step and branch from it.
- **`HotKVStore` / `AsyncHotKVStore`**: the long-term memory store. Items live across threads and
  runs, are found by namespace, filter or, with an embedding model, by meaning, and can expire.
- **No modules and no license.** They use HotKV's core data types and its vector index
  (`VSIM.*`), not RediSearch or RedisJSON-style modules, and they work on a HotKV server with or
  without an enterprise license.

Passes LangGraph's own checkpointer conformance suite (`langgraph-checkpoint-conformance`, 81
tests in 8 capabilities, including `delete_for_runs`, `copy_thread` and `prune`) with both the
sync and the async saver. Written against HotKV server 0.3.0.

## Install

```bash
pip install langgraph-checkpoint-hotkv
```

Python 3.10 or newer, `langgraph-checkpoint` 4.2 or newer.

## Checkpointer

```python
from langgraph.checkpoint.hotkv import HotKVSaver

with HotKVSaver.from_conn_string("hotkv://localhost:6379") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": "1"}})
```

With `ainvoke` and `astream` use the async saver:

```python
from langgraph.checkpoint.hotkv import AsyncHotKVSaver

async with AsyncHotKVSaver.from_conn_string("hotkv://localhost:6379") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    await graph.ainvoke(...)
```

`from_conn_string` accepts `hotkv://`, `hotkvs://` (TLS), `redis://` and `rediss://` URLs; extra
keyword arguments go to the HotKV client. To share a connection pool, build the saver yourself:
`HotKVSaver(hotkv.from_url(url))`.

Everything is stored under a key prefix (`langgraph` by default; set `prefix=`), so several
applications or environments can share one server.

### Housekeeping

```python
checkpointer.delete_thread("1")                 # forget a conversation
checkpointer.prune(["1"], strategy="keep_latest")   # keep only the newest checkpoint of a thread
checkpointer.copy_thread("1", "2")              # branch a whole thread
```

`prune` understands LangGraph's `DeltaChannel`: when a thread uses one, it keeps the older
checkpoints the channel needs to rebuild its state, so the state is never silently emptied.
Checkpoints do not expire by themselves. Prune or delete old threads on a schedule.

## Store

```python
from langgraph.store.hotkv import HotKVStore

with HotKVStore.from_conn_string("hotkv://localhost:6379") as store:
    store.put(("users", "ada"), "prefs", {"theme": "dark"})
    store.get(("users", "ada"), "prefs")
    store.search(("users",), filter={"theme": "dark"})
    store.list_namespaces(prefix=("users",))
    graph = builder.compile(checkpointer=checkpointer, store=store)
```

The filter operators are `$eq`, `$ne`, `$gt`, `$gte`, `$lt` and `$lte`. Results come newest first.

### Semantic search

Give the store an embedding model and it indexes what you put, using one HotKV vector index per
namespace:

```python
from langchain.embeddings import init_embeddings

with HotKVStore.from_conn_string(
    "hotkv://localhost:6379",
    index={"dims": 1536, "embed": init_embeddings("openai:text-embedding-3-small")},
) as s:
    s.put(("docs",), "a", {"text": "cats purr and chase mice"})
    s.search(("docs",), query="a pet that hunts", limit=3)    # results carry .score
```

`fields` (default the whole item) and a per-`put` `index=[...]` choose what is embedded;
`index=False` keeps an item out of the index. A filter is applied to the nearest neighbours the
index returns, so a very selective filter on a large namespace can return fewer results than
`limit`.

### Expiry

```python
HotKVStore.from_conn_string(url, ttl={"default_ttl": 60})     # minutes, for every new item
store.put(ns, key, value, ttl=5)                               # or per item
```

A read or search refreshes the clock unless you pass `refresh_ttl=False`.

## Things to know

- Use a standalone or replicated HotKV server. The savers and the store touch several keys in one
  transaction or script, which a cluster rejects unless every key falls in one slot; cluster mode is
  not supported in this version.
- `AsyncHotKVSaver` and `AsyncHotKVStore` also answer the synchronous methods when they are called
  from another thread than the event loop that owns the connection, which is how LangGraph runs a
  sync node of an async graph. Calling them blocking from the event loop's own thread raises
  `asyncio.InvalidStateError` instead of deadlocking.
- Item expiry is kept in whole seconds on the server, so a `ttl` of 0.02 minutes lasts 2 seconds.
  `list_namespaces` can still show a namespace whose items all expired until something searches it.
- A write to a thread that is being deleted at that moment is kept and stays listed. Writes for a
  checkpoint that was deleted in the meantime can leave a small orphan record; `delete_thread` removes
  whatever it can reach.

- Search without a `query` reads every item of the matching namespaces, which suits namespaces of
  up to a few thousand items, such as one user's memories. For larger collections use a query.
- Thread ids, checkpoint namespaces, store keys and namespace labels must not contain the control
  characters U+001E and U+001F (the store also rejects `.`, as LangGraph's other stores do).
- Values are serialised with LangGraph's own serialiser, so the usual rules apply to what a state
  may contain.
- Save a different `prefix` per environment. Deleting a prefix's keys removes its data.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/test_unit.py          # no server needed
HOTKV_TEST_URL=hotkv://127.0.0.1:6379 python -m pytest tests    # everything
```

The integration tests use a random key prefix and clean up after themselves. Run them against a
throwaway server.

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Support

- Bugs and feature requests: [GitHub issues](https://github.com/hotkv-ltd/hotkv-langgraph/issues)
- Questions and commercial support: support@hotkv.com
- Security vulnerabilities: see [SECURITY.md](SECURITY.md)

## License

Licensed under the Apache License, Version 2.0: see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Trademarks and affiliation

"HotKV" is a trademark of HotKV Ltd; the license does not grant rights to use it.
The HotKV server is a separate commercial product.

HotKV is an independent product of HotKV Ltd. It speaks the RESP protocol and implements
many Redis commands so that existing tools and client habits carry over, but it is not Redis,
Valkey, Dragonfly or KeyDB, and this package is built for and tested against HotKV. HotKV Ltd is not
affiliated with, endorsed by or sponsored by Redis Ltd., the Valkey project, DragonflyDB or KeyDB.
Redis is a registered trademark of Redis Ltd. Valkey, Dragonfly, KeyDB and all other product and
company names are trademarks of their respective owners; they are used here only to describe
protocol and command compatibility. LangGraph and LangChain are trademarks of LangChain, Inc.;
this package is an independent integration and is not affiliated with or endorsed by LangChain, Inc.
