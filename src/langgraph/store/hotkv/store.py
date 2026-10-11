# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`HotKVStore` and `AsyncHotKVStore`: LangGraph long-term memory stores backed by HotKV."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterable, Iterator, List, Optional

import hotkv
import hotkv.asyncio
from langgraph.store.base import BaseStore, IndexConfig, Op, Result, TTLConfig

from ._core import Embed, Soft, StoreCore, Tx

__all__ = ["HotKVStore", "AsyncHotKVStore"]

DEFAULT_PREFIX = "langgraph-store"


class HotKVStore(BaseStore):
    """Long-term memory for LangGraph agents in HotKV, using the synchronous client.

    ```python
    from langgraph.store.hotkv import HotKVStore

    with HotKVStore.from_conn_string("hotkv://localhost:6379") as store:
        store.put(("users", "ada"), "prefs", {"theme": "dark"})
        store.get(("users", "ada"), "prefs")
    ```

    For semantic search pass `index={"dims": 1536, "embed": embeddings}`; items
    are then found by meaning with `store.search(prefix, query="...")`. Items can
    expire: pass `ttl={"default_ttl": 60}` (minutes), or `ttl=` on a single `put`.
    """

    supports_ttl = True

    def __init__(
        self,
        client: "hotkv.HotKV",
        *,
        prefix: str = DEFAULT_PREFIX,
        index: Optional[IndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> None:
        self.client = client
        self.index_config = index
        self.ttl_config = ttl
        self._core = StoreCore(prefix, index, ttl)

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        url: str,
        *,
        prefix: str = DEFAULT_PREFIX,
        index: Optional[IndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
        **client_options: Any,
    ) -> Iterator["HotKVStore"]:
        client = hotkv.from_url(url, **client_options)
        try:
            yield cls(client, prefix=prefix, index=index, ttl=ttl)
        finally:
            client.close()

    def _exec(self, step: Any) -> Any:
        if isinstance(step, Embed):
            emb = self._core.embeddings
            return [emb.embed_query(step[0])] if step.query else emb.embed_documents(list(step))
        with self.client.pipeline(transaction=isinstance(step, Tx)) as pipe:
            for command in step:
                pipe.execute_command(*command)
            return pipe.execute(raise_on_error=not isinstance(step, Soft))

    def _run(self, flow: Any) -> Any:
        try:
            step = next(flow)
            while True:
                step = flow.send(self._exec(step))
        except StopIteration as done:
            return done.value

    def batch(self, ops: Iterable[Op]) -> List[Result]:
        return self._run(self._core.batch(list(ops)))

    async def abatch(self, ops: Iterable[Op]) -> List[Result]:
        return await asyncio.to_thread(self.batch, list(ops))


class AsyncHotKVStore(BaseStore):
    """Same as `HotKVStore`, using the asyncio client. Use the `a`-prefixed methods (`aget`, `aput`, `asearch`...)."""

    supports_ttl = True

    def __init__(
        self,
        client: "hotkv.asyncio.HotKV",
        *,
        prefix: str = DEFAULT_PREFIX,
        index: Optional[IndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
    ) -> None:
        self.client = client
        self.index_config = index
        self.ttl_config = ttl
        self._core = StoreCore(prefix, index, ttl)
        try:
            self.loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        url: str,
        *,
        prefix: str = DEFAULT_PREFIX,
        index: Optional[IndexConfig] = None,
        ttl: Optional[TTLConfig] = None,
        **client_options: Any,
    ) -> AsyncIterator["AsyncHotKVStore"]:
        client = hotkv.asyncio.from_url(url, **client_options)
        try:
            yield cls(client, prefix=prefix, index=index, ttl=ttl)
        finally:
            await client.close()

    async def _exec(self, step: Any) -> Any:
        if isinstance(step, Embed):
            emb = self._core.embeddings
            return [await emb.aembed_query(step[0])] if step.query else await emb.aembed_documents(list(step))
        async with self.client.pipeline(transaction=isinstance(step, Tx)) as pipe:
            for command in step:
                await pipe.execute_command(*command)
            return await pipe.execute(raise_on_error=not isinstance(step, Soft))

    async def _run(self, flow: Any) -> Any:
        try:
            step = next(flow)
            while True:
                step = flow.send(await self._exec(step))
        except StopIteration as done:
            return done.value

    async def abatch(self, ops: Iterable[Op]) -> List[Result]:
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        return await self._run(self._core.batch(list(ops)))

    def batch(self, ops: Iterable[Op]) -> List[Result]:
        """Synchronous access, for code running in another thread than the event loop (a sync node in an async graph)."""
        return self._sync(self.abatch, list(ops))

    # -- synchronous calls from other threads (a sync node of an async graph) ----------

    def _sync(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        loop = self.loop
        if loop is None:
            raise RuntimeError("this object has not been used from an event loop yet; use the async methods, or the synchronous class")
        try:
            if asyncio.get_running_loop() is loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls are only allowed from a different thread than the event loop that owns "
                    "the connection. Use the async methods, or the synchronous class."
                )
        except RuntimeError:
            pass  # no loop is running in this thread: the normal case
        return asyncio.run_coroutine_threadsafe(method(*args, **kwargs), loop).result()
