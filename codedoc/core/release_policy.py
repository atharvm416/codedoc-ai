"""Developer-owned capability policy for staged split-mode releases.

The policy is deliberately not configurable by users.  Public configuration
chooses ``large_file_strategy``; this module controls which implementation
milestones the installed release is allowed to expose.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SplitReleasePolicy:
    """Capabilities exposed by one release in the split patch train."""

    execution: bool
    completed_reuse: bool
    partial_recovery: bool
    node_checkpoints: bool

    def __post_init__(self) -> None:
        if self.completed_reuse and not self.execution:
            raise ValueError("split completed reuse requires split execution.")
        if self.partial_recovery and not self.execution:
            raise ValueError("split partial recovery requires split execution.")
        if self.node_checkpoints != self.partial_recovery:
            raise ValueError(
                "split node checkpointing and partial recovery must be released together."
            )


# Real single+split execution, same-path completed split reuse,
# schema-versioned node-keyed partial recovery, and node-level checkpoint
# writing are all public. An unchanged completed split file is reused with
# zero provider calls; an interrupted split file resumes only its unpaid
# leaf, reducer, and final nodes.
CURRENT_SPLIT_RELEASE = SplitReleasePolicy(
    execution=True,
    completed_reuse=True,
    partial_recovery=True,
    node_checkpoints=True,
)


def current_split_release_policy() -> SplitReleasePolicy:
    """Return the installed release's immutable developer-owned policy."""

    return CURRENT_SPLIT_RELEASE
