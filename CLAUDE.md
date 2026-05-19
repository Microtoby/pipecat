# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Changelog Fragments

Every PR with a user-facing change requires a changelog fragment at `changelog/<PR_number>.<type>.md`.

Types: `added`, `changed`, `deprecated`, `removed`, `fixed`, `performance`, `security`, `other`.

Write entries as Markdown bullet points starting with `-`. Multiple types in one PR get separate files (`1234.added.md`, `1234.fixed.md`). Multiple changes of the same type use numbered suffixes (`1234.changed.md`, `1234.changed.2.md`).

Preview the full changelog before submitting:

```bash
uv run towncrier build --draft --version Unreleased
```

## Testing Patterns

Tests use `unittest.IsolatedAsyncioTestCase`. The `run_test()` helper from `src/pipecat/tests/utils.py` wires a single processor between source/sink queues, sends frames, and returns `(downstream_frames, upstream_frames)`.

```python
class TestMyProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_something(self):
        processor = MyProcessor()
        (received_down, received_up) = await run_test(
            processor,
            frames_to_send=[TextFrame(text="hello")],
            expected_down_frames=[TextFrame],   # list of Frame classes
        )
        assert received_down[0].text == "hello"
```

Key `run_test()` params: `expected_down_frames` / `expected_up_frames` (list of Frame subclasses), `frames_to_send_direction` (default `DOWNSTREAM`), `send_end_frame` (default `True`), `enable_rtvi` (default `False`). Use `SleepFrame(sleep=N)` in `frames_to_send` to introduce delays.
