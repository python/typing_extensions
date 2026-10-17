## Context

I ran an automated audit of the open issues on this repository using the GitHub REST API (no cloning, read-only). This is a "state of the backlog" snapshot to help with triage. All metrics are derived from the current `open` state.

## Metrics (as of the latest audit run)

| Metric | Value |
|---|---|
| Open issues | 14 |
| Without labels | 11 |
| Without assignee | 14 |
| Without any reply | 5 |
| Stale (> 1 year old) | 12 |

**Top labels in use:**
- `pep-646` (2)
- `bug` (1)

## Oldest open issues (candidates for triage)

| # | Opened | Title |
|---|---|---|
| #103 | 2022-12-19 | `IntFloatTuple` from PEP 646 doesn't work as expected on Python<3.11 |
| #308 | 2023-11-29 | Multiple inheritance and ReadOnly issues? |
| #322 | 2024-01-18 | TypeError: Parameter list cannot be empty for TypeVarTuple |
| #396 | 2024-05-16 | PEP-696 specialisation tests from CPython fail |
| #443 | 2024-08-12 | Future of typing_extensions.Doc |

## Suggested actions (prioritized)

### P1 — Label taxonomy
11 out of 14 open issues have no labels. Suggest applying a minimal taxonomy:
- `bug` — defect vs. stdlib behavior
- `enhancement` — new feature proposal
- `docs` — documentation gap
- `pep-XXX` — tied to a specific PEP (existing convention: `pep-646`)
- `question` — needs clarification
- `stale` — untouched for > 12 months

### P1 — Stale issues
12 issues are > 1 year old. Suggest a `stale-bot` policy: comment after 12 months of inactivity, close after 30 more days if no response.

### P2 — Assignees
All 14 issues are unassigned. Suggest a lightweight ownership model: assign the main maintainer (JelleZijlstra) by default, or use a `triage` label to mark items awaiting review.

### P2 — Unanswered issues
5 issues have zero comments. Suggest a first-response SLA: acknowledge every new issue within 7 days.

## Automation added

I can contribute the following two files if maintainers are interested:

1. `.github/workflows/stale.yml` — a `actions/stale` workflow that marks issues as `stale` after 12 months and closes after 30 more days. Fully configurable, opt-in.
2. `.github/workflows/triage-audit.yml` — a workflow that runs the audit script weekly and uploads the report as a workflow artifact (transparency, no code changes, no issue creation).

I will paste both in follow-up comments. Happy to open a PR if maintainers prefer that path.
