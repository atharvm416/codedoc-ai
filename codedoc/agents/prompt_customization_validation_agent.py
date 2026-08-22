"""Provider-backed semantic review for active prompt customizations."""

from __future__ import annotations

from dataclasses import dataclass

from codedoc.agents.base_agent import BaseAgent
from codedoc.core.execution_model import (
    REVIEW_OWNER,
    CallManifestTracker,
    PlannedCall,
    review_call_id,
)
from codedoc.core.prompt_profiles import (
    MAX_PROFILE_SECURITY_REASONS,
    MAX_PROFILE_SECURITY_WARNINGS,
    REVIEW_SYSTEM,
    ReviewBatch,
    clean_review_verdict,
)
from codedoc.utils.errors import (
    CodeDocError,
    PromptCustomizationValidationError,
    bounded_exception_summary,
)
from codedoc.utils.json_utils import DuplicateJSONKeyError, loads_no_duplicate_keys


@dataclass(frozen=True)
class ReviewOutcome:
    verdict: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    calls_completed: int


class PromptCustomizationValidationAgent(BaseAgent):
    """Review exact deterministic batches using the configured provider."""

    agent_name = "PromptCustomizationValidationAgent"

    def __init__(self, *args, call_tracker: CallManifestTracker | None = None, **kwargs):
        super().__init__(*args, call_tracker=call_tracker, **kwargs)

    def run(self, *args, **kwargs) -> dict:  # pragma: no cover - use review_batch
        raise NotImplementedError("use review_batch()")

    def review_batch(self, batch: ReviewBatch) -> dict:
        try:
            component_ids = tuple(component.component for component in batch.components)
            planned_call = PlannedCall(
                call_id=review_call_id(
                    batch.stream_digest, component_ids, batch.ordinal
                ),
                category="prompt-review",
                owner=REVIEW_OWNER,
                ordinal=batch.ordinal,
            )
            raw = self._call_llm(
                batch.text,
                system=REVIEW_SYSTEM,
                planned_call=planned_call,
            )
            # Unlike documentation responses, a safety verdict is fail-closed:
            # markdown fences, preambles, trailing text, and multiple objects are
            # malformed rather than being repaired or extracted.
            parsed = loads_no_duplicate_keys(raw.strip())
            return clean_review_verdict(
                parsed,
                expected_review_id=batch.review_id,
                expected_batch_index=batch.ordinal,
                expected_batch_count=batch.count,
            )
        except PromptCustomizationValidationError:
            raise
        except DuplicateJSONKeyError as exc:
            raise PromptCustomizationValidationError(
                "prompt customization standards/safety review failed closed for "
                f"batch {batch.ordinal}/{batch.count}: duplicate key {exc.key!r}."
            ) from exc
        except Exception as exc:
            detail = str(exc) if isinstance(exc, CodeDocError) else bounded_exception_summary(exc)
            raise PromptCustomizationValidationError(
                "prompt customization standards/safety review failed closed for "
                f"batch {batch.ordinal}/{batch.count}: {detail}"
            ) from exc

    def review(self, batches: list[ReviewBatch]) -> ReviewOutcome:
        verdict = "SAFE"
        reasons: list[str] = []
        warnings: list[str] = []
        completed = 0
        severity = {"SAFE": 0, "RISKY": 1, "TOO_RISKY": 2}
        for batch in batches:
            cleaned = self.review_batch(batch)
            completed += 1
            if severity[cleaned["verdict"]] > severity[verdict]:
                verdict = cleaned["verdict"]
            for message in cleaned["reasons"]:
                if (
                    message not in reasons
                    and len(reasons) < MAX_PROFILE_SECURITY_REASONS
                ):
                    reasons.append(message)
            for message in cleaned["warnings"]:
                if (
                    message not in warnings
                    and len(warnings) < MAX_PROFILE_SECURITY_WARNINGS
                ):
                    warnings.append(message)
        return ReviewOutcome(verdict, tuple(reasons), tuple(warnings), completed)
