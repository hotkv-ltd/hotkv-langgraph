# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

"""LangGraph's own conformance suite, run against both savers."""

import uuid

import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.report import ProgressCallbacks

import hotkv
from langgraph.checkpoint.hotkv import AsyncHotKVSaver, HotKVSaver

from conftest import URL, wipe


class SyncPathSaver(HotKVSaver):
    """HotKVSaver whose async methods call the sync ones, so the suite exercises the sync path."""

    async def aget_tuple(self, config):
        return self.get_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id):
        self.delete_thread(thread_id)

    async def adelete_for_runs(self, run_ids):
        self.delete_for_runs(run_ids)

    async def acopy_thread(self, source_thread_id, target_thread_id):
        self.copy_thread(source_thread_id, target_thread_id)

    async def aprune(self, thread_ids, *, strategy="keep_latest"):
        self.prune(thread_ids, strategy=strategy)


def _report(report):
    lines = []
    for name, result in report.results.items():
        lines.append(f"{name}: passed={result.tests_passed} failed={result.tests_failed}")
        lines.extend(f"   {f}" for f in result.failures)
    return "\n".join(lines)


async def test_async_saver_passes_the_langgraph_conformance_suite():
    prefix = f"lgtest-{uuid.uuid4().hex[:10]}"

    @checkpointer_test(name="AsyncHotKVSaver")
    async def make():
        async with AsyncHotKVSaver.from_conn_string(URL, prefix=prefix) as saver:
            yield saver

    report = await validate(make, progress=ProgressCallbacks.quiet())
    client = hotkv.from_url(URL)
    try:
        wipe(client, prefix)
    finally:
        client.close()
    assert report.passed_all(), _report(report)
    assert report.conformance_level() == "FULL"
    assert sum(r.tests_passed for r in report.results.values()) >= 80


async def test_sync_saver_passes_the_langgraph_conformance_suite():
    prefix = f"lgtest-{uuid.uuid4().hex[:10]}"

    @checkpointer_test(name="HotKVSaver")
    async def make():
        with SyncPathSaver.from_conn_string(URL, prefix=prefix) as saver:
            yield saver

    report = await validate(make, progress=ProgressCallbacks.quiet())
    client = hotkv.from_url(URL)
    try:
        wipe(client, prefix)
    finally:
        client.close()
    assert report.passed_all(), _report(report)
    assert sum(r.tests_passed for r in report.results.values()) >= 80
