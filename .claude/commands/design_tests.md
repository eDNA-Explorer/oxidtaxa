# Design Tests

You are tasked with turning raw test research into an approved, actionable test plan. You will read a research document, prioritize the findings interactively with the user, resolve ambiguities, and produce a final test plan document.

**Input**: $ARGUMENTS (path to a research document). If no path is provided, ask the user for one before proceeding.

## Step 1: Read research

Read the file at the provided path. Extract all happy paths, edge cases, mock targets, and data transformations from every segment in the document.

## Step 2: Present findings and recommend prioritization

Organize all cases into three tiers:

**Must test:**
- Every happy path that doesn't have an existing test
- Edge cases that are likely to happen in production AND cause silent corruption or wrong results

**Should test:**
- Happy paths that have partial coverage but miss important assertions
- Edge cases that are either likely with low impact, or unlikely with high impact

**Document only:**
- Edge cases that are unlikely AND low impact
- Cases already well-covered by existing tests

Present the full tiered list to the user with reasoning for each placement. **Wait for feedback.** Adjust tiers based on user input before proceeding.

## Step 3: Discuss test approaches

For each must-test and should-test case, recommend one of:

- **Unit test (pytest)** — pure logic, no external calls, lives in `tests/`
- **Integration script** — touches external systems, lives in `scripts/`
- **Crafted-state test** — reads/writes persistent state, lives in `scripts/`

Present grouped by approach type. **Wait for feedback.** Adjust based on user input before proceeding.

## Step 4: Resolve ambiguities

For any edge case where the correct behavior is unclear, ask the user. For scope questions (too many tests? not enough?), ask. For infrastructure questions (is staging GCS available? do we have test credentials?), ask.

Do NOT proceed with unresolved questions. Every question must have an answer before moving to Step 5.

## Step 5: Write the test plan

Write to `thoughts/shared/plans/{date}-test-plan-{description}.md` where `{description}` is a short kebab-case label matching the research document. Tell the user the output file path.

Use this template:

```markdown
# Test Plan: {Feature/Area Name}

## Overview
**Date**: {YYYY-MM-DD}
**Branch**: {branch name}
**Source research**: {path to research document}
**Scope**: {what is and isn't covered}

## Summary of Changes Under Test
{1-3 sentence summary}

## Data Flow
{condensed end-to-end flow}

## What We're NOT Testing
{explicit out-of-scope items}

## Test Groups

### Group 1: {Name} [Must Test]
**Approach**: Unit test / Integration script / Crafted-state test
**Location**: `tests/{path}/test_{name}.py` or `scripts/test_{name}.py`

#### Cases:
1. **{Case name}** — Happy path | Edge case
   - **Scenario**: {what the test does}
   - **Input**: {data or how to construct it}
   - **Expected output**: {return value or side effect}
   - **Key assertion**: {single most important thing to verify}
   - **Mock targets** (if any): `patch("module.function")` -> {value}

### Group 2: {Name} [Must Test]
{same structure}

### Group N: {Name} [Should Test]
{same structure}

## Document Only (Not Implementing)
- {Case}: {reason}

## Implementation Order
1. Group {N}: {name} — {why first}
2. Group {M}: {name} — {why second}

## Shared Test Utilities
- **{Utility name}**: {what it does, which groups use it}

## Verification
### Automated:
- [ ] `poetry run pytest {path}` — unit tests pass
- [ ] `poetry run ruff format` / `ruff check` — formatting/linting
- [ ] `poetry run pyright` — type checking

### Manual:
- [ ] `poetry run python scripts/test_{name}.py` — {what it verifies}
```

## Step 6: Review and iterate

Present the plan location to the user. Invite feedback. Iterate on the plan until the user is satisfied.

---

## Guidelines

- **Be Interactive**: This is a conversation, not a monologue. Wait for user input at Steps 2, 3, 4, and 6.
- **Be Skeptical**: Question whether edge cases are realistic. Don't test things that can't happen.
- **Research Before Asking**: If you can answer a question by reading source code, do that instead of asking the user.
- **No Open Questions in Final Plan**: Every ambiguity must be resolved before writing the plan in Step 5.
- **Track Progress with TodoWrite**: Use TodoWrite to track which steps are complete.
- **Scope Ruthlessly**: Fewer well-designed tests beat many shallow ones. Push back if the test list is too long.
