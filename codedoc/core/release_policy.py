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


# Fresh release: real single+split execution is public, but every oversized split
# target is paid fresh work.  Completed reuse, partial loading, and node-level
# checkpoint writes remain unavailable until the recovery milestone.
CURRENT_SPLIT_RELEASE = SplitReleasePolicy(
    execution=True,
    completed_reuse=False,
    partial_recovery=False,
    node_checkpoints=False,
)


def current_split_release_policy() -> SplitReleasePolicy:
    """Return the installed release's immutable developer-owned policy."""

    return CURRENT_SPLIT_RELEASE
