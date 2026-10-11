# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""`HotKVSaver` and `AsyncHotKVSaver`: LangGraph checkpointers backed by HotKV."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, Tuple

import hotkv
import hotkv.asyncio
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from ._core import Core, Tx

__all__ = ["HotKVSaver", "AsyncHotKVSaver"]

DEFAULT_PREFIX = "langgraph"


class HotKVSaver(BaseCheckpointSaver[str]):
    """Stores LangGraph checkpoints in HotKV, using the synchronous client.

    ```python
    from langgraph.checkpoint.hotkv import HotKVSaver

    with HotKVSaver.from_conn_string("hotkv://localhost:6379") as saver:
        graph = builder.compile(checkpointer=saver)
    ```

    Everything is stored under `prefix`, so several applications can share one
    server. Use `AsyncHotKVSaver` for `ainvoke`/`astream`.
    """

    def __init__(
        self,
        client: "hotkv.HotKV",
        *,
        prefix: str = DEFAULT_PREFIX,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.client = client
        self._core = Core(self.serde, prefix)

    @classmethod
    @contextmanager
    def from_conn_string(cls, url: str, *, prefix: str = DEFAULT_PREFIX, **client_options: Any) -> Iterator["HotKVSaver"]:
        """Connect to `url` (`hotkv://`, `hotkvs://`, `redis://`...) and close the connection on exit."""
        client = hotkv.from_url(url, **client_options)
        try:
            yield cls(client, prefix=prefix)
        finally:
            client.close()

    # -- driving the generators -------------------------------------------------

    def _exec(self, step: Any) -> Any:
        with self.client.pipeline(transaction=isinstance(step, Tx)) as pipe:
            for command in step:
                pipe.execute_command(*command)
            return pipe.execute()

    def _run(self, flow: Any) -> Any:
        try:
            step = next(flow)
            while True:
                step = flow.send(self._exec(step))
        except StopIteration as done:
            return done.value

    # -- BaseCheckpointSaver ----------------------------------------------------

    def get_tuple(self, config: Any) -> Optional[CheckpointTuple]:
        return self._run(self._core.get_tuple(config))

    def list(
        self,
        config: Optional[Any],
        *,
        filter: Optional[dict] = None,
        before: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        yield from self._run(self._core.list(config, filter=filter, before=before, limit=limit))

    def put(self, config: Any, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> Any:
        return self._run(self._core.put(config, checkpoint, metadata, new_versions))

    def put_writes(self, config: Any, writes: Sequence[Tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        self._run(self._core.put_writes(config, writes, task_id, task_path))

    def delete_thread(self, thread_id: str) -> None:
        self._run(self._core.delete_thread(thread_id))

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        self._run(self._core.delete_for_runs(run_ids))

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self._run(self._core.copy_thread(source_thread_id, target_thread_id))

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        self._run(self._core.prune(thread_ids, strategy))


class AsyncHotKVSaver(BaseCheckpointSaver[str]):
    """Stores LangGraph checkpoints in HotKV, using the asyncio client.

    ```python
    from langgraph.checkpoint.hotkv import AsyncHotKVSaver

    async with AsyncHotKVSaver.from_conn_string("hotkv://localhost:6379") as saver:
        graph = builder.compile(checkpointer=saver)
        await graph.ainvoke(...)
    ```
    """

    def __init__(
        self,
        client: "hotkv.asyncio.HotKV",
        *,
        prefix: str = DEFAULT_PREFIX,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        self.client = client
        self._core = Core(self.serde, prefix)
        try:
            self.loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, url: str, *, prefix: str = DEFAULT_PREFIX, **client_options: Any
    ) -> AsyncIterator["AsyncHotKVSaver"]:
        """Connect to `url` and close the connection on exit."""
        client = hotkv.asyncio.from_url(url, **client_options)
        try:
            yield cls(client, prefix=prefix)
        finally:
            await client.close()

    async def _exec(self, step: Any) -> Any:
        async with self.client.pipeline(transaction=isinstance(step, Tx)) as pipe:
            for command in step:
                await pipe.execute_command(*command)
            return await pipe.execute()

    async def _run(self, flow: Any) -> Any:
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        try:
            step = next(flow)
            while True:
                step = flow.send(await self._exec(step))
        except StopIteration as done:
            return done.value

    async def aget_tuple(self, config: Any) -> Optional[CheckpointTuple]:
        return await self._run(self._core.get_tuple(config))

    async def alist(
        self,
        config: Optional[Any],
        *,
        filter: Optional[dict] = None,
        before: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in await self._run(self._core.list(config, filter=filter, before=before, limit=limit)):
            yield item

    async def aput(
        self, config: Any, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions
    ) -> Any:
        return await self._run(self._core.put(config, checkpoint, metadata, new_versions))

    async def aput_writes(
        self, config: Any, writes: Sequence[Tuple[str, Any]], task_id: str, task_path: str = ""
    ) -> None:
        await self._run(self._core.put_writes(config, writes, task_id, task_path))

    async def adelete_thread(self, thread_id: str) -> None:
        await self._run(self._core.delete_thread(thread_id))

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        await self._run(self._core.delete_for_runs(run_ids))

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        await self._run(self._core.copy_thread(source_thread_id, target_thread_id))

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        await self._run(self._core.prune(thread_ids, strategy))

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

    # The synchronous methods LangGraph calls from worker threads when a graph with sync nodes is run async.

    def get_tuple(self, config: Any) -> Optional[CheckpointTuple]:
        return self._sync(self.aget_tuple, config)

    def list(
        self,
        config: Optional[Any],
        *,
        filter: Optional[dict] = None,
        before: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        async def collect() -> list:
            return [item async for item in self.alist(config, filter=filter, before=before, limit=limit)]

        yield from self._sync(collect)

    def put(self, config: Any, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> Any:
        return self._sync(self.aput, config, checkpoint, metadata, new_versions)

    def put_writes(self, config: Any, writes: Sequence[Tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        self._sync(self.aput_writes, config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        self._sync(self.adelete_thread, thread_id)

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        self._sync(self.adelete_for_runs, run_ids)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self._sync(self.acopy_thread, source_thread_id, target_thread_id)

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        self._sync(self.aprune, thread_ids, strategy=strategy)
