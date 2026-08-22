"""Section 3A: CodeDoc-only DEBUG namespace scoping, the third-party floor,
and bounded exception rendering.

Gate 24.1 covers both halves of the privacy repair: logger levels (this file)
and the category-1 exception-rendering paths (this file plus the integration
sentinel coverage in test_progress_logging.py / test_fail_fast.py /
test_prompt_customization.py named by the plan).
"""

from __future__ import annotations

import logging

import pytest

from codedoc.utils.errors import (
    AgentError,
    ConfigError,
    ErrorReporter,
    MAX_BOUNDED_SUMMARY_CHARS,
    OutputError,
    ParseError,
    bounded_exception_summary,
)
from codedoc.utils.logger import _NOISY_LOGGERS, get_logger, set_level
from tests.support.logging_sentinels import (
    ALL_SENTINELS,
    SENTINEL_AUTHORIZATION_HEADER,
    SENTINEL_ENDPOINT_QUERY,
    SENTINEL_PROMPT_FRAGMENT,
    SENTINEL_REQUEST_BODY,
    SENTINEL_RESPONSE_BODY,
    SENTINEL_SOURCE_LINE,
    assert_no_sentinels_leaked,
    sentinel_bearing_exception,
)


@pytest.fixture(autouse=True)
def _restore_log_level():
    """Every test in this module restores codedoc to INFO afterward."""
    yield
    set_level("INFO")


# ---------------------------------------------------------------------------
# Namespace scoping: set_level touches only the codedoc logger
# ---------------------------------------------------------------------------


class TestNamespaceScoping:
    def test_set_level_debug_raises_only_codedoc_logger(self):
        root_level_before = logging.getLogger().level
        set_level("DEBUG")
        assert logging.getLogger("codedoc").level == logging.DEBUG
        # The fail-closed assertion: root is never touched by set_level, in
        # either direction.
        assert logging.getLogger().level == root_level_before

    def test_set_level_never_calls_root_setlevel(self, monkeypatch):
        calls: list[int] = []
        original = logging.Logger.setLevel

        def spy(self, level):
            if self is logging.getLogger():
                calls.append(level)
            return original(self, level)

        monkeypatch.setattr(logging.Logger, "setLevel", spy)
        set_level("DEBUG")
        set_level("INFO")
        assert calls == [], "set_level must never call logging.getLogger().setLevel(...)"

    def test_unregistered_third_party_namespace_unaffected_by_set_level(self):
        synthetic = logging.getLogger("sentinel_synthetic_sdk_9f1a2b")
        assert synthetic.name not in _NOISY_LOGGERS
        assert synthetic.level == logging.NOTSET

        effective_before = synthetic.getEffectiveLevel()
        set_level("DEBUG")
        assert synthetic.level == logging.NOTSET
        assert synthetic.getEffectiveLevel() == effective_before, (
            "an unlisted synthetic namespace must be completely unaffected by "
            "set_level(): the floor list is defense in depth, not the only "
            "barrier -- root's own level (unchanged) still governs it"
        )


# ---------------------------------------------------------------------------
# Third-party floor: exact, closed list, held at WARNING in both directions
# ---------------------------------------------------------------------------


class TestThirdPartyFloor:
    def test_configure_floors_every_listed_namespace(self):
        from codedoc.utils.logger import _configure

        _configure()
        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level >= logging.WARNING

    def test_debug_never_lowers_any_floored_namespace(self):
        set_level("DEBUG")
        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level >= logging.WARNING, (
                f"{name} must stay at WARNING-or-stricter even when codedoc is DEBUG"
            )

    def test_normal_after_verbose_keeps_floor(self):
        set_level("DEBUG")
        set_level("INFO")
        for name in _NOISY_LOGGERS:
            assert logging.getLogger(name).level >= logging.WARNING

    def test_exact_closed_floor_list(self):
        """The list is closed and exact -- every required namespace present,
        nothing extra silently relied upon."""
        assert set(_NOISY_LOGGERS) == {
            "openai",
            "anthropic",
            "google",
            "google.genai",
            "google_genai",
            "google.auth",
            "httpx",
            "httpcore",
            "urllib3",
            "requests",
        }

    @pytest.mark.parametrize(
        "descendant",
        ["httpcore.http11", "openai.lib.parsing", "google_genai._api_client"],
    )
    def test_descendant_propagation_cannot_bypass_the_floor(self, descendant):
        """A NOTSET descendant resolves to its nearest ancestor with a level
        set, so flooring the parent covers the child -- proven for the three
        namespaces the plan calls out explicitly."""
        set_level("DEBUG")
        child = logging.getLogger(descendant)
        assert child.level == logging.NOTSET
        assert child.getEffectiveLevel() >= logging.WARNING
        assert not child.isEnabledFor(logging.DEBUG)

    def test_google_and_google_underscore_are_both_required(self):
        """google.genai and google_genai are NOT the same namespace tree --
        google-genai uses the underscore form for google_genai._api_client,
        which is not a child of "google"."""
        assert "google.genai" in _NOISY_LOGGERS
        assert "google_genai" in _NOISY_LOGGERS

    def test_preconfigured_descendant_and_handler_cannot_bypass_floor(self):
        child = logging.getLogger("openai.lib.parsing.preconfigured")
        handler = logging.NullHandler(level=logging.DEBUG)
        child.setLevel(logging.DEBUG)
        child.addHandler(handler)
        try:
            set_level("DEBUG")
            assert child.level >= logging.WARNING
            assert handler.level >= logging.WARNING
            assert not child.isEnabledFor(logging.DEBUG)
        finally:
            child.removeHandler(handler)
            child.setLevel(logging.NOTSET)


# ---------------------------------------------------------------------------
# Negative sentinel scan: a third-party DEBUG record never reaches output
# ---------------------------------------------------------------------------


class TestNegativeSentinelScan:
    def test_third_party_debug_sentinel_is_absent_after_set_level_debug(self, caplog):
        set_level("DEBUG")
        third_party = logging.getLogger("openai")
        with caplog.at_level(logging.DEBUG):
            third_party.debug(
                "json_data.messages=%s prompt=%s auth=%s",
                SENTINEL_REQUEST_BODY,
                SENTINEL_PROMPT_FRAGMENT,
                SENTINEL_AUTHORIZATION_HEADER,
            )
        assert_no_sentinels_leaked(caplog.text)

    def test_codedoc_debug_still_emits_its_own_records(self, caplog):
        set_level("DEBUG")
        mine = get_logger("codedoc.unit.test_logging_isolation")
        with caplog.at_level(logging.DEBUG, logger="codedoc.unit.test_logging_isolation"):
            mine.debug("bounded codedoc diagnostic: file_count=3")
        assert "bounded codedoc diagnostic" in caplog.text


# ---------------------------------------------------------------------------
# bounded_exception_summary: the frozen contract
# ---------------------------------------------------------------------------


class TestBoundedExceptionSummary:
    def test_returns_reason_code_only_when_no_detail(self):
        assert bounded_exception_summary(RuntimeError("anything")) == "unknown-error"

    def test_unmapped_type_is_fail_closed_unknown_error(self):
        class WeirdError(Exception):
            pass

        summary = bounded_exception_summary(WeirdError(SENTINEL_SOURCE_LINE))
        assert summary == "unknown-error"
        assert_no_sentinels_leaked(summary)

    def test_unmapped_type_cannot_smuggle_structured_numeric_detail(self):
        class WeirdError(Exception):
            pass

        exc = WeirdError(SENTINEL_SOURCE_LINE)
        exc.status_code = 429
        exc.code = 401
        exc.retry_after = 60
        assert bounded_exception_summary(exc) == "unknown-error"

    def test_os_error_maps_to_filesystem_error_with_category_detail(self):
        summary = bounded_exception_summary(PermissionError(SENTINEL_SOURCE_LINE))
        assert summary == "filesystem-error (permission denied)"
        assert_no_sentinels_leaked(summary)

    def test_cancelled_error_maps_to_cancelled(self):
        import concurrent.futures

        assert bounded_exception_summary(concurrent.futures.CancelledError()) == "cancelled"
        assert bounded_exception_summary(KeyboardInterrupt()) == "cancelled"
        assert bounded_exception_summary(SystemExit()) == "cancelled"

    @pytest.mark.parametrize(
        "module, name, expected_code",
        [
            ("openai", "AuthenticationError", "provider-authentication-rejected"),
            ("openai", "RateLimitError", "provider-rate-limited"),
            ("openai", "APITimeoutError", "provider-timeout"),
            ("openai", "APIConnectionError", "provider-connection-failed"),
            ("openai", "NotFoundError", "provider-model-unavailable"),
            ("openai", "UnprocessableEntityError", "provider-response-malformed"),
            ("anthropic", "AuthenticationError", "provider-authentication-rejected"),
            ("anthropic", "RateLimitError", "provider-rate-limited"),
            ("google.genai.errors", "ClientError", "provider-request-failed"),
            ("google.genai.errors", "ServerError", "provider-request-failed"),
        ],
    )
    def test_provider_sdk_type_mapping_by_module_and_name(self, module, name, expected_code):
        fake_cls = type(name, (Exception,), {"__module__": module})
        exc = fake_cls(SENTINEL_RESPONSE_BODY)
        summary = bounded_exception_summary(exc)
        assert summary == expected_code
        assert_no_sentinels_leaked(summary)

    def test_subclass_of_a_mapped_type_still_matches_via_mro(self):
        base = type("APIError", (Exception,), {"__module__": "openai"})
        derived = type("SomeFutureSubclass", (base,), {"__module__": "openai"})
        assert bounded_exception_summary(derived("x")) == "provider-request-failed"

    def test_status_code_attribute_becomes_the_bounded_detail(self):
        fake_cls = type("RateLimitError", (Exception,), {"__module__": "openai"})
        exc = fake_cls("boom")
        exc.status_code = 429
        assert bounded_exception_summary(exc) == "provider-rate-limited (429)"

    def test_genai_code_attribute_becomes_the_bounded_detail(self):
        fake_cls = type("ClientError", (Exception,), {"__module__": "google.genai.errors"})
        exc = fake_cls("boom")
        exc.code = 401
        assert bounded_exception_summary(exc) == "provider-request-failed (401)"

    def test_never_reads_message_body_response_or_headers(self):
        class Hostile(Exception):
            pass

        exc = Hostile("safe text")
        exc.message = SENTINEL_PROMPT_FRAGMENT
        exc.body = SENTINEL_REQUEST_BODY
        exc.response = SENTINEL_RESPONSE_BODY
        exc.request = SENTINEL_ENDPOINT_QUERY
        exc.headers = {"Authorization": SENTINEL_AUTHORIZATION_HEADER}
        exc.text = SENTINEL_SOURCE_LINE
        exc.content = SENTINEL_SOURCE_LINE
        summary = bounded_exception_summary(exc)
        assert summary == "unknown-error"
        assert_no_sentinels_leaked(summary)

    def test_never_traverses_cause_or_context(self):
        cause = RuntimeError(SENTINEL_SOURCE_LINE)
        try:
            raise RuntimeError("outer") from cause
        except RuntimeError as exc:
            summary = bounded_exception_summary(exc)
        assert_no_sentinels_leaked(summary)

    def test_truncated_to_max_bounded_summary_chars(self):
        fake_cls = type("RateLimitError", (Exception,), {"__module__": "openai"})
        exc = fake_cls("boom")
        exc.status_code = 12345678901234567890  # absurdly long int detail
        summary = bounded_exception_summary(exc)
        assert len(summary) <= MAX_BOUNDED_SUMMARY_CHARS

    def test_kitchen_sink_sentinel_exception_never_leaks(self):
        summary = bounded_exception_summary(sentinel_bearing_exception())
        assert_no_sentinels_leaked(summary)
        assert summary == "unknown-error"


# ---------------------------------------------------------------------------
# ErrorReporter: two-tier rendering boundary, no retained traceback
# ---------------------------------------------------------------------------


class TestErrorReporterTwoTier:
    def test_codedoc_error_renders_unchanged(self):
        reporter = ErrorReporter()
        exc = ParseError("owner/file.py", "line 12: sentinel syntax detail")
        reporter.record(exc)
        entry = reporter.summary()
        assert "line 12: sentinel syntax detail" in entry
        assert "owner/file.py" in entry

    def test_config_error_renders_unchanged(self):
        reporter = ErrorReporter()
        exc = ConfigError("codedoc.config.json: duplicate key 'foo' at line 3")
        reporter.record(exc)
        assert "codedoc.config.json" in reporter.summary()
        assert "line 3" in reporter.summary()

    def test_output_error_renders_unchanged(self):
        reporter = ErrorReporter()
        exc = OutputError("out/codedoc.json", "permission denied (PermissionError, errno 13)")
        reporter.record(exc)
        assert "permission denied" in reporter.summary()

    def test_foreign_exception_is_bounded(self):
        reporter = ErrorReporter()
        reporter.record(sentinel_bearing_exception("boom"))
        assert_no_sentinels_leaked(reporter.summary())
        assert "unknown-error" in reporter.summary()

    def test_agent_error_wrapping_a_sentinel_bearing_cause_is_bounded(self):
        reporter = ErrorReporter()
        cause = sentinel_bearing_exception()
        exc = AgentError("DocumentationAgent", "m.py", bounded_exception_summary(cause))
        reporter.record(exc)
        assert_no_sentinels_leaked(reporter.summary())

    def test_no_entry_ever_carries_a_traceback_key(self):
        reporter = ErrorReporter()
        try:
            raise sentinel_bearing_exception()
        except Exception as exc:
            reporter.record(exc)
        # Access the private entries only to prove the field is gone; no
        # public accessor exposes it because none should exist.
        assert all("traceback" not in entry for entry in reporter._entries)

    def test_warning_level_entries_excluded_from_summary(self):
        reporter = ErrorReporter()
        reporter.record(RuntimeError("recovered"), level="warning")
        assert reporter.summary() == ""
        assert reporter.has_issues()
        assert not reporter.has_errors()


# ---------------------------------------------------------------------------
# Category 2: local deterministic errors are asserted PRESENT and UNCHANGED
# ---------------------------------------------------------------------------


class TestCategoryTwoPreserved:
    """A test that redacts these has over-applied the contract (TRAPS #1)."""

    def test_parse_error_keeps_file_and_line(self):
        exc = ParseError("src/bad.py", "invalid syntax (src/bad.py, line 7)")
        assert "line 7" in str(exc)
        assert "src/bad.py" in str(exc)

    def test_config_error_keeps_filename_and_json_position(self):
        exc = ConfigError("'codedoc.config.json' is not valid JSON: line 4 column 1 (char 55)")
        assert "codedoc.config.json" in str(exc)
        assert "line 4 column 1" in str(exc)


def test_all_sentinel_classes_are_distinct_and_nonempty():
    assert len(set(ALL_SENTINELS)) == len(ALL_SENTINELS)
    assert all(isinstance(s, str) and s for s in ALL_SENTINELS)


# ---------------------------------------------------------------------------
# Windows redirected-logging Unicode safety (platform-neutral unit coverage;
# the genuine PowerShell subprocess reproduction lives in
# tests/platform/logging/test_redirected_verbose.py, Windows-only)
# ---------------------------------------------------------------------------


class TestStreamEncodingSafety:
    def test_reconfigures_error_handler_to_backslashreplace(self):
        from codedoc.utils.logger import _make_stream_encoding_safe

        calls: list[dict] = []

        class _FakeStream:
            def reconfigure(self, **kwargs):
                calls.append(kwargs)

        _make_stream_encoding_safe(_FakeStream())
        assert calls == [{"errors": "backslashreplace"}]

    def test_silently_skips_a_stream_without_reconfigure(self):
        from codedoc.utils.logger import _make_stream_encoding_safe

        class _NoReconfigure:
            pass

        _make_stream_encoding_safe(_NoReconfigure())  # must not raise

    def test_silently_skips_a_stream_whose_reconfigure_rejects_the_call(self):
        from codedoc.utils.logger import _make_stream_encoding_safe

        class _RefusingStream:
            def reconfigure(self, **kwargs):
                raise ValueError("not supported")

        _make_stream_encoding_safe(_RefusingStream())  # must not raise

    def test_unencodable_character_becomes_a_deterministic_escape_not_a_crash(self, tmp_path):
        """Direct proof of the P2 mechanism without a subprocess: writing a
        character absent from cp1252 through a strict-mode stream raises;
        the same stream after _make_stream_encoding_safe() renders a
        deterministic backslash-escaped sequence instead."""
        from codedoc.utils.logger import _make_stream_encoding_safe

        path = tmp_path / "legacy_codepage.txt"
        with open(path, "w", encoding="cp1252") as fh:
            with pytest.raises(UnicodeEncodeError):
                fh.write("→")

        path2 = tmp_path / "legacy_codepage_safe.txt"
        with open(path2, "w", encoding="cp1252") as fh:
            _make_stream_encoding_safe(fh)
            fh.write("→")  # must not raise
        assert path2.read_text(encoding="cp1252") == "\\u2192"
