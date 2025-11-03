# Contributing Guide

How to set up, code, test, review, and release so contributions meet our Definition of Done.

## Code of Conduct

*Reference the project/community behavior expectations and reporting process.*

All members must act respectfully and respond within agreed timelines. Disagreements are resolved through discussion and votes. Any communication issues or unprofessional behavior should be reported to the project lead or partner contact for review.

## Getting Started

*List prerequisites, setup steps, environment variables/secrets handling, and how to run the app locally.*

Install Conda and required CUDA/NCCL drivers. Clone the repo, create a clean environment using the provided lockfile, and confirm tests run locally. Configure credentials and secrets using environment variables, not plain files. Run the setup script to install dependencies.

## Branching & Workflow

*Describe the workflow (e.g., trunk-based or GitFlow), default branch, branch naming, and when to rebase vs. merge.*

Use trunk-based development. The main branch is protected and only updated through pull requests. Feature branches follow the pattern feature/\<short-description\>. Rebase often and squash commits before merging. Merge only after review approvals and passing CI.

## Issues & Planning

*Explain how to file issues, required templates/labels, estimation, and triage/assignment practices.*

Open issues for all new work using the standard issue template. Label with type (bug, feature, doc) and include clear acceptance criteria. Assign yourself and set due dates matching the sprint schedule. Levi tracks progress and updates task lists weekly.

## Commit Messages

*State the convention (e.g., Conventional Commits), include examples, and how to reference issues.*

Follow the Conventional Commits format. Example: feat(hpc): add new NCCL configuration check. Reference the related issue number in each commit. Keep messages short and meaningful to make change tracking clear.

## Code Style, Linting & Formatting

*Name the formatter/linter, config file locations, and the exact commands to check/fix locally.*

Use the configured linter and formatter before submitting a pull request. Run the local script lint.sh to check and fix style errors. No code with warnings or type errors may be merged. Formatting follows Python PEP8 and project style guides.

## Testing

*Define required test types, how to run tests, expected coverage thresholds, and when new/updated tests are mandatory.*

All pull requests must include or update unit and integration tests. Coverage should not drop below 90 percent. Use pytest to run all tests locally before pushing. HPC and experiment scripts should include dry-run checks to ensure configurations are valid.

## Pull Requests & Reviews

*Outline PR requirements (template, checklist, size limits), reviewer expectations, approval rules, and required status checks.*

Pull requests must include a clear description, linked issue, and all relevant artifacts such as metrics, configs, and logs. Two reviewers are required: one technical and one domain or documentation reviewer. All review comments must be resolved before merge.

## CI/CD

*Link to pipeline definitions, list mandatory jobs, how to view logs/re-run jobs, and what must pass before merge/release.*

All merges run through continuous integration. Jobs include linting, type checks, tests, and reproducibility checks. CI must be fully green before merge. Logs are available on the project’s CI dashboard. Do not bypass CI for any reason.

## Security & Secrets

*State how to report vulnerabilities, prohibited patterns (hard-coded secrets), dependency update policy, and scanning tools.*

Do not commit secrets or access tokens. Store them as environment variables. Report any suspected vulnerability or security issue to Caleb immediately. Dependencies are checked and updated each sprint to reduce risk from outdated packages.

## Documentation Expectations

*Specify what must be updated (README, docs/, API refs, CHANGELOG) and docstring/comment standards.*

Every change that affects workflows or environment setup must be documented in the README or runbook. Marat oversees document consistency. All experiment results, configurations, and reproducibility details are logged with correct references and dates.

## Release Process

*Describe versioning scheme, tagging, changelog generation, packaging/publishing steps, and rollback process.*

At the end of each sprint, tag the release using vX.Y-sprintN format. Include lockfiles, metrics, and plots. Update the changelog with new results and fixes. Verify reproducibility by rerunning baseline tests before final tagging.

## 

## Support & Contact

*Provide a maintainer contact channel, expected response windows, and where to ask questions.*

For technical issues contact Caleb. For partner or mentor coordination contact Aidan. For documentation or process concerns contact Marat. Response is expected within 24 hours on workdays through Discord or text if urgent.