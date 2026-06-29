"""Provider-backed single-to-triple prompt-profile routing agent (0.11.1).

Workstream D.  When triple mode is selected but only a *customized* ``single``
structure is configured, CodeDoc makes exactly one paid routing call (after the
ordinary source security review) to propose three explicit triple structures.

The agent is a thin, fail-closed wrapper around the configured provider: it
renders the deterministic routing request, bounds the raw response before
parsing, parses it as one bare JSON object rejecting duplicate keys at any depth,
and runs the full version-2 deterministic validator over the proposed structures.
It never writes files, never mutates config, and never relaxes validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from codedoc.agents.base_agent import BaseAgent
from codedoc.core.prompt_profiles import (
    MAX_PROMPT_PROFILE_ROUTING_RESPONSE_CHARS,
    ROUTING_SYSTEM,
    AgentProfile,
    build_routing_request,
    validate_routing_response,
)
from codedoc.utils.errors import PromptCustomizationValidationError
from codedoc.utils.json_utils import DuplicateJSONKeyError, loads_no_duplicate_keys


@dataclass(frozen=True)
class RoutingOutcome:
    triple: Mapping[str, AgentProfile]
    factors: tuple[str, ...]
    calls_attempted: int
    calls_completed: int


class PromptProfileRoutingAgent(BaseAgent):
    """Route a validated single structure into explicit triple structures."""

    agent_name = "PromptProfileRoutingAgent"

    def run(self, *args, **kwargs) -> dict:  # pragma: no cover - use route()
        raise NotImplementedError("use route()")

    def route(self, single_profile: AgentProfile, conversion_id: str) -> RoutingOutcome:
        # Deterministic preflight: rebuild the bounded request (identical bytes to
        # the pre-provider size proof).  Raises locally if it would exceed the
        # routing ceiling, before any provider attempt is counted.
        request = build_routing_request(single_profile, conversion_id)
        attempted = 0
        try:
            attempted = 1
            raw = self._call_llm(request, system=ROUTING_SYSTEM)
        except PromptCustomizationValidationError:
            raise
        except Exception as exc:
            raise PromptCustomizationValidationError(
                "single-to-triple prompt-profile routing failed closed: "
                f"{exc}"
            ) from exc
        if len(raw) > MAX_PROMPT_PROFILE_ROUTING_RESPONSE_CHARS:
            raise PromptCustomizationValidationError(
                "prompt profile: the routing response is "
                f"{len(raw)} characters, exceeding the "
                f"{MAX_PROMPT_PROFILE_ROUTING_RESPONSE_CHARS}-character ceiling; "
                "failing closed before parsing."
            )
        try:
            # Fail-closed like the safety verdict: fences, preambles, trailing
            # text, multiple objects, and duplicate keys are malformed.
            parsed = loads_no_duplicate_keys(raw.strip())
        except DuplicateJSONKeyError as exc:
            raise PromptCustomizationValidationError(
                f"prompt profile: routing response had a duplicate key {exc.key!r}."
            ) from exc
        except json.JSONDecodeError as exc:
            raise PromptCustomizationValidationError(
                f"prompt profile: routing response was not one JSON object: {exc}."
            ) from exc
        validated = validate_routing_response(
            parsed, expected_conversion_id=conversion_id
        )
        return RoutingOutcome(
            triple=validated["triple"],
            factors=validated["factors"],
            calls_attempted=attempted,
            calls_completed=1,
        )
