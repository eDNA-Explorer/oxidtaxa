# Research Tests

You are tasked with analyzing the current branch's changes to discover what needs testing. Do NOT write any tests — your job is to produce a complete analysis document with all findings consolidated into a single file.

## Step 1: Understand what changed and how data flows through it

Run `git diff develop...HEAD --stat` and `git log develop..HEAD --oneline` to see the full scope of changes on this branch. Read every changed file.

### 1a. Inventory every changed function

For each new or modified function, record:
- What inputs it receives and what it returns
- What external systems it calls (DB, API, cloud storage, queue)
- What persistent state it reads or writes (files, manifests, database rows, caches)

### 1b. Trace the data flow end-to-end

Draw the path data takes through the changed code, from entry point to final output. Write it as a numbered sequence:

```
1. Caller invokes function_a(raw_input)
2. function_a validates input, calls external_api() → returns ResponseObj
3. function_a transforms ResponseObj into InternalModel, passes to function_b()
4. function_b writes InternalModel to GCS as JSON
5. function_c reads JSON from GCS, aggregates across records
6. function_c writes summary to DB, returns AggregateResult
```

For each handoff between functions, note:
- **What is passed**: the exact type and shape (dataclass, dict, list of dicts, DataFrame)
- **What could go wrong at this boundary**: type mismatch, missing fields, empty collection, stale data from a previous step
- **Where data is transformed**: any place where the shape, type, or meaning of the data changes (these are high-value test targets)

## Step 2: Check what's already tested

Search for existing tests that cover the changed code:
- Look in the corresponding `tests/` directory for the changed modules
- Check if the existing tests still pass: `poetry run pytest <relevant test path> -v`
- Note which functions have tests and which don't

## Step 3: Spawn parallel agents to analyze segments of the data flow

Using the data flow from Step 1b, divide the flow into logical segments. A segment is a coherent chunk of the pipeline — it might be a single function, or it might span multiple functions across multiple files that together handle one phase of the work (e.g., "cache resolution and download", "state sync and manifest persistence", "export submission and task tracking").

Good segment boundaries are where:
- The data shape changes (raw input → validated model → aggregated result)
- An external system is called (DB read → transform → GCS write)
- Responsibility shifts (submission logic → monitoring logic → resubmission logic)

For each segment, spawn a separate `codebase-analyzer` agent. Give each agent:
- The list of files and functions in its segment
- The data flow steps it owns (e.g., "steps 3-5 from the flow: transform → write to GCS → read back and aggregate")
- The handoff points at the boundaries (what comes in from the previous segment, what goes out to the next)
- A **unique output file** to write findings to:

```
thoughts/shared/research/{date}-test-analysis-{segment-name}.md
```

**CRITICAL: Context management.** Each agent's prompt MUST include these instructions:
> Write ALL of your findings to the output file specified above. When you are done, respond with ONLY the file path you wrote to and a one-line summary (e.g., "Wrote 8 happy path cases and 12 edge cases to thoughts/shared/research/2026-02-16-test-analysis-cache-resolution.md"). Do NOT include the file contents in your response. The parent will read the file directly.

This prevents agent responses from flooding the parent context and hitting the context limit.

Each agent must:

1. Read all source code in its segment — every file, not just the changed functions
2. Identify **happy path** test cases — the normal flow through the segment with representative inputs, including how data transforms at each step
3. Identify **edge cases** by running through this question list for every step and handoff in the segment:
   - What if the input is **empty or zero**?
   - What if a counter or limit is at its **boundary** (max, min, exactly 1)?
   - What if stored state is **stale or wrong** (cached value contradicts reality)?
   - What if the operation **partially fails** (some items succeed, some fail)?
   - What if the external system returns **something unexpected** (unknown status, empty, timeout)?
   - What if the input is **duplicated** (same entry twice, operation runs twice)?
   - What if a field's value **contradicts another field** (timestamp set but status says "pending")?
   - What if this thing is **already done** and gets processed again?
   - What if there's **nothing to do** (zero items, empty list, no changes)?
4. If there are status fields or lifecycle stages, enumerate every state transition and ask: what if the event fires twice? What if it arrives in the wrong state?
5. **Write findings to the output file** using this format:

```markdown
# Test Analysis: {segment name}

## Segment overview
- **Flow steps covered**: steps N-M from the data flow
- **Entry point**: what data enters this segment (type, shape, source)
- **Exit point**: what data leaves this segment (type, shape, destination)
- **Files involved**:
  - `path/to/file.py` — what role it plays in this segment

## Data transformations
For each place data changes shape, type, or meaning within the segment:
- **Where**: `function_name()` in `file.py`, line ~N
- **Input shape**: what comes in (e.g., `list[dict]` with keys x, y, z)
- **Output shape**: what comes out (e.g., `DataFrame` with columns a, b)
- **What could break**: missing keys, empty list, type mismatch, null values

## Happy path cases
For each case:
- **Scenario**: describe the normal flow through this segment
- **Input**: representative data entering the segment
- **Expected output**: what should come out the other end
- **Key steps**: which functions are called in sequence
- **Side effects**: state written, external calls made
- **Existing test coverage**: yes/no, and where

## Edge cases found
For each case:
- **Description**: what the edge case is
- **Where in the flow**: which step or handoff is affected
- **Trigger**: how this could happen in production
- **Expected behavior**: what the code should do
- **Actual behavior**: what you believe the code currently does (note if uncertain)
- **Impact if wrong**: silent corruption / loud crash / data loss / cosmetic
- **Suggested test approach**: unit test / mock helper / crafted-state

## Mock targets identified
For each external dependency that would need mocking:
- **Called by**: which function in the segment calls it
- **Import style**: top-level or lazy/local
- **Patch target**: exact string for `@patch("...")`
- **Realistic return value**: what the mock should return (type and shape)
```

Launch all agents in parallel. Each one focuses on its own segment and writes to its own file.

## Step 4: Consolidate and write final document

After all agents complete:

1. Read all individual agent output files from `thoughts/shared/research/{date}-test-analysis-*.md`
2. Merge everything into a single comprehensive document at `thoughts/shared/research/{date}-test-research-{description}.md` where `{description}` is a short kebab-case label for the feature/area (e.g., `cache-resolution`, `export-pipeline`)
3. The document contains ALL findings from all segments — the full inventory of happy paths, edge cases, mock targets, data transformations — organized by segment but in one file
4. Delete the individual agent temp files after consolidation
5. Tell the user the output file path

**Final output format** (the single consolidated document):

```markdown
# Test Research: {description}

## Branch Changes
{git diff summary, files changed, commit log}

## Data Flow
{end-to-end numbered data flow from Step 1b}

## Existing Test Coverage
{what's already tested from Step 2}

## Segment: {segment-1 name}
### Overview
- Flow steps covered, entry/exit points, files involved
### Data Transformations
- Each transformation with input/output shapes and what could break
### Happy Path Cases
- Full details per the existing template (scenario, input, expected output, key steps, side effects, existing coverage)
### Edge Cases
- Full details per the existing template (description, where in flow, trigger, expected behavior, actual behavior, impact, suggested approach)
### Mock Targets
- Each target with called-by, import style, patch target, realistic return value

## Segment: {segment-2 name}
{same structure}

## Segment: {segment-N name}
{same structure}
```

## Quick Reference: The 10 Edge Case Questions

For every step and handoff in the data flow, ask:

1. What if the input is **empty or zero**?
2. What if a counter or limit is at its **boundary**?
3. What if stored state is **stale or wrong**?
4. What if the operation **partially fails**?
5. What if the external system returns **something unexpected**?
6. What if the input is **duplicated**?
7. What if **versions don't match**?
8. What if a field's value **contradicts another field**?
9. What if this thing is **already done** and gets processed again?
10. What if there's **nothing to do** (zero items, no changes)?
