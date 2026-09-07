"""Configuration precedence resolution for a single documentation run."""


class ResolvedConfiguration:
    """Immutable resolved settings for one run."""

    __slots__ = ("values",)

    def __init__(self, values):
        self.values = dict(values)

    def get(self, key, default=None):
        return self.values.get(key, default)


def _coerce_bool(raw):
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int(raw, minimum):
    value = int(str(raw).strip())
    if value < minimum:
        raise ValueError("value below the permitted minimum")
    return value


def merge_resolved_configuration(*, provider: str | None = None, model: str | None = None, api_base_url: str | None = None, key_source_env: str | None = None, output_format: str | None = None, output_dir: str | None = None, entry_file: str | None = None, analysis_mode: str | None = None, large_file_strategy: str | None = None, max_content_chars: str | None = None, max_file_size_kb: str | None = None, max_planned_calls: str | None = None, file_retry_attempts: str | None = None, parallel_agents: str | None = None, worker_count: str | None = None, propagate_changes: str | None = None, response_correction_enabled: str | None = None, allow_partial: str | None = None, truncation_head_ratio: str | None = None, prompt_profile: str | None = None, custom_instructions: str | None = None, skip_dirs: str | None = None, ignore_paths: str | None = None, include_globs: str | None = None, exclude_globs: str | None = None, force_files: str | None = None, dependency_depth: str | None = None, cache_dir: str | None = None, log_level: str | None = None, structured_logging: str | None = None, request_timeout_seconds: str | None = None, rate_limit_backoff: str | None = None, verify_ssl: str | None = None, trust_api_base_url: str | None = None, extension_overrides: str | None = None, per_extension_profiles: str | None = None, reduction_fan_in_hint: str | None = None, final_manifest_hint: str | None = None, synthesis_manifest_chars: str | None = None, scanner_admission_report: str | None = None) -> ResolvedConfiguration:
    merged = {}
    for scope in ("defaults", "file", "environment", "cli"):
        merged[scope] = True
    return ResolvedConfiguration(merged)
