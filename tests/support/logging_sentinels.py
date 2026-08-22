"""Shared sentinel values for logging-privacy and bounded-exception tests.

Every sentinel below is a distinctive, greppable string standing in for a
category of sensitive content (raw source, prompt text, request/response
body, endpoint query/fragment data, credentials).  Section 3A requires that
none of these ever reach console output, captured logs, exceptions,
artifacts, or recovery state; :func:`assert_no_sentinels_leaked` is the one
shared negative-scan helper every test in this family uses so the checked set
never drifts between test modules.

Consumed by ``tests/unit/utils/test_logging_isolation.py``,
``tests/platform/logging/test_redirected_verbose.py``, and the verbose
split-diagnostics coverage in ``tests/integration/pipeline/test_progress_logging.py``.
"""

from __future__ import annotations

SENTINEL_SOURCE_LINE = "def sentinel_source_marker_9f3a2b(): return 'SENTINEL_SOURCE_BODY_TEXT'"
SENTINEL_PROMPT_FRAGMENT = "SENTINEL_PROMPT_INSTRUCTION_7c1d4e_do_not_leak_this_prompt"
SENTINEL_REQUEST_BODY = '{"messages": [{"role": "user", "content": "SENTINEL_REQUEST_BODY_5e6f1a"}]}'
SENTINEL_RESPONSE_BODY = '{"choices": [{"message": {"content": "SENTINEL_RESPONSE_BODY_2b8d9c"}}]}'
SENTINEL_ENDPOINT_QUERY = "https://api.example-sentinel.test/v1/chat?api_key=SENTINEL_QUERY_KEY_4a7e1f"
SENTINEL_ENDPOINT_FRAGMENT = "https://api.example-sentinel.test/v1/chat#token=SENTINEL_FRAGMENT_TOKEN_1d9b3c"
SENTINEL_AUTHORIZATION_HEADER = "Bearer SENTINEL_BEARER_TOKEN_3c5a8e9d"
SENTINEL_OPENAI_API_KEY = "sk-SENTINEL0123456789abcdefKEYFAKE"
SENTINEL_ANTHROPIC_API_KEY = "sk-ant-SENTINEL0123456789abcdefKEYFAKE"
SENTINEL_GOOGLE_API_KEY = "AIzaSENTINELFAKEKEY0123456789abc"

# Every sentinel that must never appear in a bounded/public surface.
ALL_SENTINELS: tuple[str, ...] = (
    SENTINEL_SOURCE_LINE,
    SENTINEL_PROMPT_FRAGMENT,
    SENTINEL_REQUEST_BODY,
    SENTINEL_RESPONSE_BODY,
    SENTINEL_ENDPOINT_QUERY,
    SENTINEL_ENDPOINT_FRAGMENT,
    SENTINEL_AUTHORIZATION_HEADER,
    SENTINEL_OPENAI_API_KEY,
    SENTINEL_ANTHROPIC_API_KEY,
    SENTINEL_GOOGLE_API_KEY,
)


def assert_no_sentinels_leaked(*blobs: str) -> None:
    """Assert that none of :data:`ALL_SENTINELS` appear in any of *blobs*.

    *blobs* may be console/log text, JSON, Markdown, or recovery-file text.
    Reports every sentinel that leaked (not just the first) so one failed
    assertion names every offending value.
    """
    leaked: list[str] = []
    for blob in blobs:
        if not blob:
            continue
        for sentinel in ALL_SENTINELS:
            if sentinel in blob:
                leaked.append(sentinel)
    assert not leaked, f"sentinel value(s) leaked into bounded output: {sorted(set(leaked))!r}"


def sentinel_bearing_message(prefix: str = "") -> str:
    """One message embedding every sentinel class at once, space-separated."""
    parts = [prefix] if prefix else []
    parts.extend(
        [
            f"source={SENTINEL_SOURCE_LINE}",
            f"prompt={SENTINEL_PROMPT_FRAGMENT}",
            f"request_body={SENTINEL_REQUEST_BODY}",
            f"response_body={SENTINEL_RESPONSE_BODY}",
            f"endpoint_query={SENTINEL_ENDPOINT_QUERY}",
            f"endpoint_fragment={SENTINEL_ENDPOINT_FRAGMENT}",
            f"authorization={SENTINEL_AUTHORIZATION_HEADER}",
            f"api_key={SENTINEL_OPENAI_API_KEY}",
        ]
    )
    return " ".join(parts)


def sentinel_bearing_exception(prefix: str = "") -> Exception:
    """Build one exception whose message embeds every sentinel class at once.

    Used to drive the "one sentinel test per category-1 path" requirement:
    inject this, then assert none of its embedded sentinel substrings survive
    whatever bounded rendering path is under test.
    """
    return RuntimeError(sentinel_bearing_message(prefix))
