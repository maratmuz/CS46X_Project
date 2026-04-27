---
layout: default
title: Contributing Guide
---

# Contributing Guide

[← Back to Home](index.md)

How to set up, code, test, review, and release so contributions meet our Definition of Done.

---

## Code of Conduct

All members must act respectfully and respond within agreed timelines. Disagreements are resolved through discussion and votes. Communication issues or unprofessional behavior should be reported to the project lead or partner contact for review.

---

## Getting Started

1. Install Conda and required CUDA/NCCL drivers
2. Clone the repo and `cd` into the project root
3. Load CUDA modules (on OSU HPC):
   ```bash
   module load cuda/12.8
   module load cudnn/8.9_cuda12
   ```
4. Run the setup script:
   ```bash
   ./scripts/setup/setup_evo2_conda.sh
   ```
5. Confirm tests pass: `pytest`

Configure credentials and secrets using **environment variables only** — never commit them to files.

---

## Branching & Workflow

We use **trunk-based development**.

- `main` is protected — only updated through pull requests
- Feature branches: `feature/<short-description>`
- Rebase often and squash commits before merging
- Merge only after review approvals and passing CI

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
feat(hpc): add new NCCL configuration check
fix(seq2expr): correct tokenizer padding logic
docs(readme): update CUDA version requirements
```

Reference the related issue number in each commit.

---

## Code Style

- Python: **PEP8**, enforced via linter
- Run `./scripts/lint.sh` before opening a PR
- No warnings or type errors may be merged

---

## Testing

- All PRs must include or update unit and integration tests
- Coverage must not drop below **90%**
- Run locally with `pytest` before pushing
- HPC scripts should include dry-run checks

---

## Pull Requests

PRs must include:

- Clear description and linked issue
- Relevant artifacts (metrics, configs, logs)
- Passing CI (lint, type check, tests, reproducibility)
- **Two reviewers**: one technical, one domain/docs

All review comments must be resolved before merge.

---

## Security & Secrets

- Never commit secrets, tokens, or credentials
- Store all sensitive values as environment variables
- Report suspected vulnerabilities to Caleb immediately
- Dependencies are reviewed and updated each sprint

---

## Documentation

Every change affecting workflows or environment setup must be reflected in the README or relevant doc. Marat oversees document consistency.

---

## Release Process

At the end of each sprint:

1. Tag release: `vX.Y-sprintN`
2. Include lockfiles, metrics, and plots
3. Update the changelog
4. Verify reproducibility by rerunning baseline tests

---

## Contact

| Area | Contact | Channel |
|------|---------|---------|
| Technical issues | Caleb Lowe | Discord / text |
| Partner/mentor coordination | Aiden Gabriel | Discord / text |
| Documentation/process | Marat Muzaffarov | Discord / text |
| Project tracking | Levi Minch | Discord / text |

Response expected within **24 hours** on workdays.
