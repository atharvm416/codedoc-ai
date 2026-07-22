"""Tests organized by feature ownership."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from codedoc.core.db import compute_file_hash, read_source_snapshot, read_source_text
from codedoc.core.execution_model import (
    REVIEW_OWNER,
    AgentCallContext,
    CallManifest,
    CallManifestTracker,
    FileExecutionRequest,
    PlannedCall,
    _validate_manifest,
    build_call_manifest,
    documentation_call_id,
    review_call_id,
)
from codedoc.core.prompt_profiles import (
    FileScope,
    ResolvedFileShapeBundle,
    ResolvedProfile,
    ReviewBatch,
    ReviewComponent,
)
from codedoc.parser.factory import parse_file, parse_source


def _bundle(basename: str = "mod.py"):
    return ResolvedProfile("single", None).resolve_bundle(FileScope(basename=basename))


def _context(**overrides) -> AgentCallContext:
    defaults = dict(
        analysis_mode="single",
        max_content_chars=12000,
        truncation_head_ratio=0.70,
        resolved_shape_bundle=_bundle(),
    )
    defaults.update(overrides)
    return AgentCallContext(**defaults)


def _request(tmp_path: Path, rel_path: str = "mod.py", content: str = "x = 1\n") -> FileExecutionRequest:
    absolute = tmp_path / rel_path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_text(content, encoding="utf-8", newline="")
    content_hash, decoded = read_source_snapshot(absolute)
    return FileExecutionRequest(
        rel_path=rel_path,
        language="python",
        imports=tuple(parse_source({"path": absolute, "language": "python"}, decoded)),
        content=decoded,
        content_hash=content_hash,
        context=_context(),
    )


def _batch(ordinal=1, count=1, stream_digest="digest-a", component_ids=("single/combined/*",)):
    components = tuple(
        ReviewComponent(component=cid, block_text="block", fields=(("description", "string"),))
        for cid in component_ids
    )
    return ReviewBatch(
        ordinal=ordinal,
        count=count,
        review_id="rev-test",
        stream_digest=stream_digest,
        unit_total=1,
        components=components,
        units=(),
        text="batch text",
    )


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_agent_call_context_is_frozen(self):
        ctx = _context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.max_content_chars = 999

    def test_file_execution_request_is_frozen(self, tmp_path):
        request = _request(tmp_path)
        with pytest.raises(dataclasses.FrozenInstanceError):
            request.content = "y = 2\n"

    def test_planned_call_is_frozen(self):
        call = PlannedCall(call_id="x", category="file-documentation", owner="a.py", ordinal=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            call.ordinal = 2

    def test_call_manifest_is_frozen(self):
        manifest = CallManifest(calls=(), digest="d")
        with pytest.raises(dataclasses.FrozenInstanceError):
            manifest.digest = "other"


# ---------------------------------------------------------------------------
# Explicit constructor contracts (not implications inferred from annotations)
# ---------------------------------------------------------------------------


class TestConstructorContracts:
    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"analysis_mode": "quadruple"}, "unsupported analysis mode"),
            ({"max_content_chars": 0}, "greater than zero"),
            ({"max_content_chars": -1}, "greater than zero"),
            ({"truncation_head_ratio": 0.0}, "between zero and one"),
            ({"truncation_head_ratio": 1.0}, "between zero and one"),
        ],
    )
    def test_agent_call_context_rejects_invalid_values(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            _context(**overrides)

    def test_agent_call_context_rejects_a_bundle_from_another_mode(self):
        triple_bundle = ResolvedProfile("triple", None).resolve_bundle(
            FileScope(basename="mod.py")
        )
        with pytest.raises(ValueError, match="does not match analysis mode"):
            _context(analysis_mode="single", resolved_shape_bundle=triple_bundle)

    def test_agent_call_context_bundle_selections_are_read_only(self):
        ctx = _context()
        with pytest.raises(TypeError):
            ctx.resolved_shape_bundle.selections["combined"] = None
        # The caller's own mutable mapping cannot reach into the frozen context.
        mutable = dict(_bundle().selections)
        held = _context(
            resolved_shape_bundle=ResolvedFileShapeBundle(
                mode="single",
                scope=FileScope(basename="mod.py"),
                selections=mutable,
                digest=_bundle().digest,
            )
        )
        mutable.clear()
        assert set(held.resolved_shape_bundle.selections) == {"combined"}

    @pytest.mark.parametrize("rel_path", ["", ".", "/abs/mod.py", "../mod.py", "a/../../b.py"])
    def test_file_execution_request_rejects_non_project_relative_paths(
        self, tmp_path, rel_path
    ):
        with pytest.raises(ValueError, match="rel_path must"):
            FileExecutionRequest(
                rel_path=rel_path,
                language="python",
                imports=(),
                content="x = 1\n",
                content_hash="hash",
                context=_context(),
            )

    def test_file_execution_request_does_not_carry_a_source_path(self):
        assert "absolute_path" not in {
            field.name for field in dataclasses.fields(FileExecutionRequest)
        }

    def test_file_execution_request_copies_imports_into_a_tuple(self, tmp_path):
        supplied = ["os", "sys"]
        request = FileExecutionRequest(
            rel_path="mod.py",
            language="python",
            imports=supplied,
            content="x = 1\n",
            content_hash="hash",
            context=_context(),
        )
        supplied.append("late")
        assert request.imports == ("os", "sys")


# ---------------------------------------------------------------------------
# Raw-byte hash / decoded-snapshot agreement
# ---------------------------------------------------------------------------


class TestSourceSnapshot:
    def test_ordinary_ascii_matches_existing_helpers(self, tmp_path):
        path = tmp_path / "plain.py"
        path.write_text("x = 1\ny = 2\n", encoding="utf-8", newline="")

        content_hash, content = read_source_snapshot(path)

        assert content_hash == compute_file_hash(path)
        assert content == read_source_text(path)

    def test_bom_bearing_file_hashes_raw_bytes_and_strips_bom_on_decode(self, tmp_path):
        path = tmp_path / "bom.py"
        raw = b"\xef\xbb\xbfx = 1\n"
        path.write_bytes(raw)

        content_hash, content = read_source_snapshot(path)

        assert content_hash == hashlib.sha256(raw).hexdigest()
        assert content_hash == compute_file_hash(path)
        assert not content.startswith("﻿")
        assert content == "x = 1\n"
        # The hash is over the BOM-bearing raw bytes, not the stripped text.
        assert content_hash != hashlib.sha256(content.encode("utf-8")).hexdigest()

    def test_invalid_utf8_file_hashes_raw_bytes_and_replaces_on_decode(self, tmp_path):
        path = tmp_path / "invalid.py"
        raw = b"x = '\xff\xfe'\n"
        path.write_bytes(raw)

        content_hash, content = read_source_snapshot(path)

        assert content_hash == hashlib.sha256(raw).hexdigest()
        assert content_hash == compute_file_hash(path)
        assert "�" in content
        # Never reconstructed by re-encoding the (lossy) decoded string.
        assert content_hash != hashlib.sha256(content.encode("utf-8")).hexdigest()

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (b"import os\r\nVALUE = 1\r\n", "import os\nVALUE = 1\n"),
            (b"import os\rVALUE = 1\r", "import os\nVALUE = 1\n"),
            (b"import os\r\nVALUE = 1\n", "import os\nVALUE = 1\n"),
        ],
        ids=["crlf", "cr", "mixed"],
    )
    def test_line_endings_are_translated_exactly_as_the_canonical_text_read(
        self, tmp_path, raw, expected
    ):
        """The canonical text policy is ``Path.read_text()``, which translates
        universal newlines.  Snapshot content must stay character-identical to
        it: prompt bytes, the ``len(content)`` truncation decision, and the
        ``_max_context_revision`` cache-identity key all derive from this text,
        so an untranslated decode would silently change prompts and invalidate
        existing records for CRLF sources."""
        path = tmp_path / "endings.py"
        path.write_bytes(raw)

        content_hash, content = read_source_snapshot(path)

        assert content == expected
        assert content == read_source_text(path)
        assert len(content) == len(read_source_text(path))
        # The hash still describes the untranslated raw bytes.
        assert content_hash == hashlib.sha256(raw).hexdigest()
        assert content_hash == compute_file_hash(path)


# ---------------------------------------------------------------------------
# Snapshot-derived imports and no worker reread
# ---------------------------------------------------------------------------


class TestRequestConstruction:
    def test_snapshot_derived_imports_match_path_based_parse(self, tmp_path):
        path = tmp_path / "mod.py"
        path.write_text("import os\nfrom .sibling import helper\n", encoding="utf-8", newline="")
        descriptor = {"path": path, "rel_path": "mod.py", "language": "python", "extension": ".py"}

        _, content = read_source_snapshot(path)
        via_snapshot = parse_source(descriptor, content)
        via_path = parse_file(descriptor)

        assert via_snapshot == via_path == ["os", ".sibling"]

    @pytest.mark.parametrize(
        "raw", [b"import os\r\nfrom .sib import h\r\n", b"import os\rfrom .sib import h\r"]
    )
    def test_snapshot_and_path_routes_agree_for_non_lf_line_endings(self, tmp_path, raw):
        path = tmp_path / "mod.py"
        path.write_bytes(raw)
        descriptor = {"path": path, "rel_path": "mod.py", "language": "python", "extension": ".py"}

        _, content = read_source_snapshot(path)

        assert parse_source(descriptor, content) == parse_file(descriptor) == ["os", ".sib"]

    def test_request_does_not_require_the_source_file_to_still_exist(self, tmp_path):
        request = _request(tmp_path, content="import os\n")
        # The request stores no path, so the test holds the backing file itself.
        (tmp_path / "mod.py").unlink()

        assert request.content == "import os\n"
        assert request.imports == ("os",)
        assert request.content_hash == hashlib.sha256(b"import os\n").hexdigest()

    def test_request_normalizes_windows_relative_paths(self, tmp_path):
        request = _request(tmp_path, rel_path="pkg\\mod.py")
        assert request.rel_path == "pkg/mod.py"

    def test_request_rejects_parent_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="project-relative"):
            FileExecutionRequest(
                rel_path="../mod.py",
                language="python",
                imports=(),
                content="x = 1\n",
                content_hash="hash",
                context=_context(),
            )


# ---------------------------------------------------------------------------
# Deterministic, category-specific, domain-separated call IDs
# ---------------------------------------------------------------------------


class TestCallIds:
    def test_documentation_call_id_is_deterministic(self):
        first = documentation_call_id("a.py", "single", "combined", 1)
        second = documentation_call_id("a.py", "single", "combined", 1)
        assert first == second
        assert len(first) == 64  # sha256 hex digest

    def test_documentation_call_id_differs_by_each_field(self):
        base = documentation_call_id("a.py", "single", "combined", 1)
        assert base != documentation_call_id("b.py", "single", "combined", 1)
        assert base != documentation_call_id("a.py", "triple", "combined", 1)
        assert base != documentation_call_id("a.py", "single", "structure", 1)
        assert base != documentation_call_id("a.py", "single", "combined", 2)

    def test_review_call_id_is_deterministic_and_domain_separated_from_documentation(self):
        review_id = review_call_id("digest-a", ("single/combined/*",), 1)
        assert review_id == review_call_id("digest-a", ("single/combined/*",), 1)
        # No accidental collision between the two categories' id spaces.
        assert review_id != documentation_call_id("digest-a", "single/combined/*", "x", 1)

    def test_call_id_does_not_collide_across_field_boundaries(self):
        """Domain separation: a naive join of ("ab","c") and ("a","bc") must not collide."""
        left = documentation_call_id("ab", "c", "combined", 1)
        right = documentation_call_id("a", "bc", "combined", 1)
        assert left != right


# ---------------------------------------------------------------------------
# Manifest ordering, digest, duplicate rejection, and bounds
# ---------------------------------------------------------------------------


class TestCallManifest:
    def test_review_calls_precede_documentation_calls_in_batch_order(self):
        batches = [_batch(ordinal=1, count=2), _batch(ordinal=2, count=2)]
        manifest = build_call_manifest(batches, ["b.py", "a.py"], "single")

        categories = [call.category for call in manifest.calls]
        assert categories == ["prompt-review", "prompt-review", "file-documentation", "file-documentation"]
        review_calls = [c for c in manifest.calls if c.category == "prompt-review"]
        assert [c.ordinal for c in review_calls] == [1, 2]
        assert all(c.owner == REVIEW_OWNER for c in review_calls)

    def test_documentation_calls_sorted_by_rel_path_then_canonical_agent_order(self):
        manifest = build_call_manifest([], ["b.py", "a.py"], "triple")
        doc_calls = [c for c in manifest.calls if c.category == "file-documentation"]

        owners_in_order = [c.owner for c in doc_calls]
        assert owners_in_order == ["a.py", "a.py", "a.py", "b.py", "b.py", "b.py"]
        assert [c.ordinal for c in doc_calls[:3]] == [1, 2, 3]

    def test_single_mode_has_one_ordinal_per_file(self):
        manifest = build_call_manifest([], ["a.py"], "single")
        assert len(manifest.calls) == 1
        assert manifest.calls[0].ordinal == 1

    def test_manifest_digest_is_deterministic_and_order_sensitive(self):
        manifest_a = build_call_manifest([], ["a.py", "b.py"], "single")
        manifest_b = build_call_manifest([], ["a.py", "b.py"], "single")
        manifest_c = build_call_manifest([], ["a.py", "b.py", "c.py"], "single")

        assert manifest_a.digest == manifest_b.digest
        assert manifest_a.digest != manifest_c.digest
        assert manifest_a.digest == hashlib.sha256(
            "\n".join(c.call_id for c in manifest_a.calls).encode("utf-8")
        ).hexdigest()

    def test_empty_manifest_is_valid_and_has_a_stable_digest(self):
        manifest = build_call_manifest([], [], "single")
        assert manifest.calls == ()
        assert manifest.digest == hashlib.sha256(b"").hexdigest()

    def test_duplicate_call_id_is_rejected(self):
        calls = [
            PlannedCall(call_id="dup", category="file-documentation", owner="a.py", ordinal=1),
            PlannedCall(call_id="dup", category="file-documentation", owner="b.py", ordinal=1),
        ]
        with pytest.raises(ValueError, match="duplicate"):
            _validate_manifest(calls, expected_total=2)

    def test_unknown_category_is_rejected(self):
        calls = [PlannedCall(call_id="x", category="bogus", owner="a.py", ordinal=1)]
        with pytest.raises(ValueError, match="unknown"):
            _validate_manifest(calls, expected_total=1)

    def test_non_consecutive_ordinals_for_one_owner_are_rejected(self):
        calls = [
            PlannedCall(call_id="a", category="file-documentation", owner="f.py", ordinal=1),
            PlannedCall(call_id="b", category="file-documentation", owner="f.py", ordinal=3),
        ]
        with pytest.raises(ValueError, match="consecutive"):
            _validate_manifest(calls, expected_total=2)

    def test_total_length_bound_is_enforced(self):
        calls = [PlannedCall(call_id="a", category="file-documentation", owner="f.py", ordinal=1)]
        with pytest.raises(ValueError, match="expected"):
            _validate_manifest(calls, expected_total=2)


class TestCallManifestTracker:
    def test_initial_consumption_and_additional_attachment(self):
        manifest = build_call_manifest([], ["a.py"], "single")
        tracker = CallManifestTracker(manifest)
        call = manifest.calls[0]

        tracker.authorize(call, additional_attempt=False)
        tracker.authorize(call, additional_attempt=True)

        assert tracker.snapshot() == {
            "attempted_logical_calls": 1,
            "planned_calls_not_attempted": 0,
        }

    def test_double_initial_consumption_is_an_internal_defect(self):
        manifest = build_call_manifest([], ["a.py"], "single")
        tracker = CallManifestTracker(manifest)
        call = manifest.calls[0]
        tracker.authorize(call, additional_attempt=False)

        with pytest.raises(RuntimeError, match="more than once"):
            tracker.authorize(call, additional_attempt=False)

    def test_additional_attempt_requires_an_attempted_origin(self):
        manifest = build_call_manifest([], ["a.py"], "single")
        tracker = CallManifestTracker(manifest)

        with pytest.raises(RuntimeError, match="no attempted logical call"):
            tracker.authorize(manifest.calls[0], additional_attempt=True)

    def test_unknown_and_mismatched_calls_are_rejected(self):
        manifest = build_call_manifest([], ["a.py"], "single")
        tracker = CallManifestTracker(manifest)
        expected = manifest.calls[0]
        unknown = dataclasses.replace(expected, call_id="unknown")
        mismatch = dataclasses.replace(expected, owner="b.py")

        with pytest.raises(RuntimeError, match="unknown"):
            tracker.authorize(unknown, additional_attempt=False)
        with pytest.raises(RuntimeError, match="metadata"):
            tracker.authorize(mismatch, additional_attempt=False)
