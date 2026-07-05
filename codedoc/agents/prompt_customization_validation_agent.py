"""Provider-backed semantic review for active prompt customizations."""

from __future__ import annotations

from dataclasses import dataclass

from codedoc.agents.base_agent import BaseAgent
from codedoc.core.prompt_profiles import (
    MAX_PROFILE_SECURITY_REASONS,
    MAX_PROFILE_SECURITY_WARNINGS,
    REVIEW_SYSTEM,
    ReviewBatch,
    clean_review_verdict,
)
from codedoc.utils.errors import PromptCustomizationValidationError
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

    def run(self, *args, **kwargs) -> dict:  # pragma: no cover - use review_batch
        raise NotImplementedError("use review_batch()")

    def review_batch(self, batch: ReviewBatch) -> dict:
        try:
            raw = self._call_llm(batch.text, system=REVIEW_SYSTEM)
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
            raise PromptCustomizationValidationError(
                "prompt customization standards/safety review failed closed for "
                f"batch {batch.ordinal}/{batch.count}: {exc}"
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
