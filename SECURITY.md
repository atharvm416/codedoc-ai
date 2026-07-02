# Security Policy

## Supported Versions

Security fixes are handled for the latest released version of `codedoc-ai`.

## Reporting a Vulnerability

Please do not open a public issue for secrets exposure, prompt injection risks, arbitrary file access, or other security-sensitive reports.

Until a dedicated security contact is published, report vulnerabilities privately to the repository owner through GitHub.

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

During a run, `crash_recovery.json` may contain generated documentation metadata.
Treat it and final JSON/Markdown output according to the sensitivity of the source
repository. Legacy databases, checkpoints, build files, issue logs, external
profiles, and managed `.gitignore` files are outside CodeDoc's runtime file
contract and are not discovered or modified.
