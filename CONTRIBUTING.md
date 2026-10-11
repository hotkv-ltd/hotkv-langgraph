# Contributing to the HotKV LangGraph Integration

Thank you for your interest in improving this package. Bug reports, fixes and
documentation improvements are welcome.

## Reporting problems

- Bugs and feature requests: open an issue on the
  [issue tracker](https://github.com/hotkv-ltd/hotkv-langgraph/issues) and fill in
  the template. Include the langgraph-checkpoint-hotkv version, your Python and LangGraph versions,
  the HotKV server version and a minimal way to reproduce the problem.
- Security vulnerabilities: do not open a public issue. Follow
  [SECURITY.md](SECURITY.md).

## Making a change

1. For anything larger than a small fix, open an issue first so the approach can
   be agreed before you spend time on it.
2. Fork the repository and create a branch from `main`.
3. Keep the change focused. Storage logic lives once in `_core.py` of the checkpointer and of the
   store, and the sync and async classes only drive it, so a change there covers both.
4. Add or update tests for the behavior you change.
5. Run the checks below and make sure they pass.
6. Open a pull request against `main` and describe what changed and why.

## Checks

These need no server:

```bash
pip install -e ".[dev]"
python -m pytest tests/test_unit.py
```

The integration tests, including LangGraph's checkpointer conformance suite, need a running HotKV
server and are skipped unless `HOTKV_TEST_URL` is set. They use a random key prefix, but run them
only against a throwaway server.

## Source headers

Every source file starts with the project header:

```
Copyright (c) 2026 HotKV Ltd
Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
```

## License of contributions

This project is licensed under the Apache License, Version 2.0. By submitting a
contribution you agree that it is licensed under the same terms (section 5 of
the license).

## Conduct

Be respectful and constructive. Harassment and personal attacks are not
tolerated, and maintainers may remove comments or contributions that cross
that line.
