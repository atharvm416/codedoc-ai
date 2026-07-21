"""Immutable execution model: frozen per-file requests and the call manifest.

This module owns the frozen, provider-neutral types that carry already-resolved
planning decisions into execution:

- :class:`AgentCallContext` / :class:`FileExecutionRequest` — one immutable,
  self-contained request per provider-bound file. ``codedoc.core.planning``
  constructs these on the provider-free path (see
  :func:`codedoc.core.db.read_source_snapshot`); workers only read them.
- :class:`PlannedCall` / :class:`CallManifest` — the canonical, ordered set of
  initially planned logical calls shared by dry-run and real execution.

No provider, agent, or worker constructs these types itself; that would be a
second planner. Retries and corrections are actual usage, not initially
planned calls, and never consume or add a manifest entry.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, Sequence

from codedoc.core.prompt_profiles import ResolvedFileShapeBundle, ReviewBatch

_ID_SEP = "\x1f"


@dataclass(frozen=True, slots=True)
class AgentCallContext:
    """Immutable per-call context.

    ``analysis_mode``, ``max_content_chars``, and ``truncation_head_ratio`` are
    identical across every context built within one run; only
    ``resolved_shape_bundle`` varies per file (extension scope).
    """

    analysis_mode: str
    max_content_chars: int
    truncation_head_ratio: float
    resolved_shape_bundle: ResolvedFileShapeBundle

    def __post_init__(self) -> None:
        if self.analysis_mode not in DOC_AGENTS_BY_MODE:
            raise ValueError(f"unsupported analysis mode: {self.analysis_mode!r}")
        if self.max_content_chars <= 0:
            raise ValueError("max_content_chars must be greater than zero.")
        if not 0 < self.truncation_head_ratio < 1:
            raise ValueError("truncation_head_ratio must be between zero and one.")
        bundle = self.resolved_shape_bundle
        if bundle.mode != self.analysis_mode:
            raise ValueError("resolved shape bundle mode does not match analysis mode.")
        immutable_bundle = ResolvedFileShapeBundle(
            mode=bundle.mode,
            scope=bundle.scope,
            selections=MappingProxyType(dict(bundle.selections)),
            digest=bundle.digest,
        )
        object.__setattr__(self, "resolved_shape_bundle", immutable_bundle)


@dataclass(frozen=True, slots=True)
class FileExecutionRequest:
    """One frozen, self-contained request for a provider-bound file.

    ``content_hash`` is the SHA-256 of the raw file bytes; ``content`` is those
    same bytes decoded with the canonical ``utf-8-sig`` / ``errors="replace"``
    policy — see :func:`codedoc.core.db.read_source_snapshot`, which produces
    both from one read. ``imports`` is derived from that identical decoded
    snapshot via :func:`codedoc.parser.factory.parse_source`. A worker consumes
    this object only; it never reopens the source file or recomputes any of
    these fields.
    """

    rel_path: str
    absolute_path: Path
    language: str
    imports: tuple[str, ...]
    content: str
    content_hash: str
    context: AgentCallContext

    def __post_init__(self) -> None:
        normalized = _normalize_rel_path(self.rel_path)
        object.__setattr__(self, "rel_path", normalized)
        absolute_path = Path(self.absolute_path)
        if not absolute_path.is_absolute():
            raise ValueError("absolute_path must be absolute.")
        object.__setattr__(self, "absolute_path", absolute_path)
        object.__setattr__(self, "imports", tuple(self.imports))


PlannedCallCategory = Literal["prompt-review", "file-documentation"]


@dataclass(frozen=True, slots=True)
class PlannedCall:
    """One initially planned logical call.

    ``call_id`` identifies an initially planned logical call, not retries or
    corrections. ``owner`` groups calls that share a consecutive ordinal
    sequence — a file's relative path for documentation calls, or the fixed
    review-owner sentinel for prompt-review calls.
    """

    call_id: str
    category: PlannedCallCategory
    owner: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class CallManifest:
    """The canonical, ordered set of initially planned calls for one run."""

    calls: tuple[PlannedCall, ...]
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "calls", tuple(self.calls))


def _normalize_rel_path(rel_path: str) -> str:
    """Return one normalized project-relative POSIX path.

    Backslashes are accepted at construction boundaries for Windows callers,
    but absolute paths, empty paths, and parent traversal are rejected.  Every
    stored request and every documentation call ID therefore uses the same
    canonical path spelling.
    """
    raw = str(rel_path).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if not raw or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"rel_path must be a project-relative path: {rel_path!r}")
    normalized = candidate.as_posix()
    if normalized in ("", "."):
        raise ValueError(f"rel_path must name a file: {rel_path!r}")
    return normalized


# ---------------------------------------------------------------------------
# Domain-separated call identifiers
# ---------------------------------------------------------------------------


def _domain_call_id(category: str, *parts: str) -> str:
    """Domain-separated SHA-256 identifying one initially planned logical call.

    Fields are joined with an ASCII unit-separator so that, e.g.,
    ``("ab", "c")`` and ``("a", "bc")`` never collide.
    """
    payload = _ID_SEP.join((category, *parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def documentation_call_id(
    rel_path: str, analysis_mode: str, agent: str, ordinal: int
) -> str:
    """The call id for one initial per-file documentation call."""
    return _domain_call_id(
        "file-documentation",
        _normalize_rel_path(rel_path),
        analysis_mode,
        agent,
        str(ordinal),
    )


def review_call_id(
    stream_digest: str, component_ids: Sequence[str], ordinal: int
) -> str:
    """The call id for one prompt-customization review batch."""
    return _domain_call_id("prompt-review", stream_digest, *component_ids, str(ordinal))


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------

# The fixed per-owner sentinel for every prompt-review call. Review batches
# have no natural per-file owner; they share one consecutive ordinal sequence
# for the whole run.
REVIEW_OWNER = "prompt-review"

# The canonical, fixed per-mode agent order, identical to
# ``codedoc.core.prompt_profiles.VALID_AGENTS_BY_MODE`` — documentation-call
# agent naming and ordinal order never drift from prompt-profile component
# naming because both are defined from the same registry shape.
DOC_AGENTS_BY_MODE: dict[str, tuple[str, ...]] = {
    "single": ("combined",),
    "triple": ("structure", "dependency", "documentation"),
}


def build_call_manifest(
    review_batches: Sequence[ReviewBatch],
    agent_rels: Sequence[str],
    analysis_mode: str,
) -> CallManifest:
    """Build and validate the one canonical call manifest for a run.

    Review calls come first, in canonical review-batch order; documentation
    calls follow, sorted by relative path and then by the canonical per-mode
    agent order. Every initially planned logical call maps to exactly one
    manifest entry; retries and corrections are excluded and never consume or
    add a manifest slot.
    """
    agents = DOC_AGENTS_BY_MODE[analysis_mode]
    calls: list[PlannedCall] = []

    for batch in review_batches:
        component_ids = tuple(component.component for component in batch.components)
        calls.append(
            PlannedCall(
                call_id=review_call_id(batch.stream_digest, component_ids, batch.ordinal),
                category="prompt-review",
                owner=REVIEW_OWNER,
                ordinal=batch.ordinal,
            )
        )

    for rel_path in sorted(agent_rels):
        for ordinal, agent in enumerate(agents, start=1):
            calls.append(
                PlannedCall(
                    call_id=documentation_call_id(rel_path, analysis_mode, agent, ordinal),
                    category="file-documentation",
                    owner=rel_path,
                    ordinal=ordinal,
                )
            )

    expected_total = len(review_batches) + len(agent_rels) * len(agents)
    _validate_manifest(calls, expected_total=expected_total)

    digest = hashlib.sha256(
        "\n".join(call.call_id for call in calls).encode("utf-8")
    ).hexdigest()
    return CallManifest(calls=tuple(calls), digest=digest)


def _validate_manifest(calls: list[PlannedCall], *, expected_total: int) -> None:
    """Internal defect check: unique ids, known categories, consecutive
    per-owner ordinals starting at 1, and the expected total length."""
    if len(calls) != expected_total:
        raise ValueError(
            f"call manifest has {len(calls)} entries; expected {expected_total}."
        )

    seen_ids: set[str] = set()
    per_owner_ordinals: dict[tuple[str, str], list[int]] = {}
    for call in calls:
        if call.call_id in seen_ids:
            raise ValueError(f"duplicate planned call id: {call.call_id}")
        seen_ids.add(call.call_id)
        if call.category not in ("prompt-review", "file-documentation"):
            raise ValueError(f"unknown planned-call category: {call.category!r}")
        per_owner_ordinals.setdefault((call.category, call.owner), []).append(
            call.ordinal
        )

    for (category, owner), ordinals in per_owner_ordinals.items():
        expected = list(range(1, len(ordinals) + 1))
        if ordinals != expected:
            raise ValueError(
                f"planned calls for {category} owner {owner!r} are not consecutive ordinals "
                f"starting at 1: {ordinals}"
            )


class CallManifestTracker:
    """Thread-safe consumption state for one immutable call manifest.

    Initial provider attempts consume a planned logical ID exactly once.
    Retries and corrections must attach to an ID that has already been
    consumed; they never consume another manifest entry.  The tracker stores no
    prompts, source, responses, credentials, or token/accounting data.
    """

    def __init__(self, manifest: CallManifest) -> None:
        self._calls = {call.call_id: call for call in manifest.calls}
        if len(self._calls) != len(manifest.calls):
            raise ValueError("call manifest contains duplicate call IDs.")
        self._attempted: set[str] = set()
        self._lock = threading.Lock()

    def authorize(self, call: PlannedCall, *, additional_attempt: bool) -> None:
        """Authorize one provider attempt for *call*.

        ``additional_attempt=False`` consumes the planned logical call.
        ``True`` attaches a retry/correction to an already-consumed call.
        Unknown IDs, metadata mismatches, double initial consumption, and an
        additional attempt without an originating initial attempt are internal
        defects and fail before the provider is called.
        """
        with self._lock:
            expected = self._calls.get(call.call_id)
            if expected is None:
                raise RuntimeError(
                    f"provider attempt referenced unknown planned call ID {call.call_id}."
                )
            if expected != call:
                raise RuntimeError(
                    "provider attempt metadata does not match its planned call ID."
                )
            already_attempted = call.call_id in self._attempted
            if additional_attempt:
                if not already_attempted:
                    raise RuntimeError(
                        "additional provider attempt has no attempted logical call."
                    )
                return
            if already_attempted:
                raise RuntimeError(
                    f"planned call ID {call.call_id} was consumed more than once."
                )
            self._attempted.add(call.call_id)

    def owner_was_attempted(self, category: PlannedCallCategory, owner: str) -> bool:
        """Whether any planned logical call for ``(category, owner)`` ran."""
        with self._lock:
            return any(
                call_id in self._attempted
                and call.category == category
                and call.owner == owner
                for call_id, call in self._calls.items()
            )

    def snapshot(self) -> dict[str, int]:
        """Return provider-free planned/attempted reconciliation counts."""
        with self._lock:
            attempted = len(self._attempted)
            total = len(self._calls)
        return {
            "attempted_logical_calls": attempted,
            "planned_calls_not_attempted": total - attempted,
        }
