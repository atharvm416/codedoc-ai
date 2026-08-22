# Security Policy

## Supported Versions

Security fixes are handled for the latest released version of `codedoc-ai`.

## Reporting a Vulnerability

Please do not open a public issue for secrets exposure, prompt injection risks, arbitrary file access, or other security-sensitive reports.

Use GitHub's private vulnerability reporting form:
https://github.com/atharvm416/codedoc-ai/security/advisories/new

When reporting, include:

- A clear description of the issue.
- Steps to reproduce.
- Impact.
- Any suggested fix.

## Secret Handling

Never commit API keys, local LLM credentials, or generated documentation that
contains private source information. CodeDoc reads credentials from operating-
system environment variables and never auto-loads `.env` files.

Runtime configuration and instruction customization come only from the exact
`codedoc.config.json`. Active custom instructions are deterministically validated
and provider-reviewed before persistent mutation; `TOO_RISKY` cannot be bypassed.

During a run, `crash_recovery.json` contains generated documentation metadata
and may contain sensitive derived project information. Treat it, verbose logs,
and final JSON/Markdown output according to the sensitivity of the source
repository. Current verbose mode emits bounded CodeDoc diagnostics and never
enables raw provider or transport request logging, but a dependency or older
release may have violated that boundary.

Keep sensitive reproduction evidence private and retain only what diagnosis
requires. Moving an incomplete recovery file or log aside preserves evidence
for a matching version or private investigation. Removing it through your
normal sensitive-data process is an explicit discard; CodeDoc does not claim
that ordinary file deletion is secure erasure.

Legacy databases, checkpoints, build files, issue logs, external profiles, and
managed `.gitignore` files are outside CodeDoc's runtime file contract and are
not discovered or modified.
