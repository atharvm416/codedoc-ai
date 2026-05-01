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

Never commit `.env` files, API keys, local LLM credentials, generated docs containing private code, or `codedoc_db.json` if it includes private project metadata.
