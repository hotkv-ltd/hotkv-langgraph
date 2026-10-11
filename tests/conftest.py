# Copyright (c) 2026 HotKV Ltd
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

import os
import uuid

import pytest

import hotkv

URL = os.environ.get("HOTKV_TEST_URL")


def pytest_collection_modifyitems(config, items):
    if URL:
        return
    skip = pytest.mark.skip(reason="set HOTKV_TEST_URL to a HotKV server to run these tests")
    for item in items:
        if "test_unit" not in item.nodeid:
            item.add_marker(skip)


def wipe(client, prefix: str) -> None:
    """Delete every key and vector index under `prefix` (tests use a random one)."""
    for name in client.vsim_info():
        if name.decode().startswith(f"{prefix}:"):
            client.vsim_dropindex(name)
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor, match=f"{prefix}:*", count=500)
        if keys:
            client.unlink(*keys)
        if cursor == 0:
            return


@pytest.fixture
def prefix():
    p = f"lgtest-{uuid.uuid4().hex[:10]}"
    yield p
    client = hotkv.from_url(URL)
    try:
        wipe(client, p)
    finally:
        client.close()
