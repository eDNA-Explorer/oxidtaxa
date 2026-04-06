# Debug

You are tasked with helping debug issues during development or testing. This command allows you to investigate problems by examining logs, pipeline state, and git history without editing files.

## Initial Response

When invoked WITH a context file:
```
I'll help debug issues with [file name]. Let me understand the current state.

What specific problem are you encountering?
- What were you trying to test/implement?
- What went wrong?
- Any error messages?

I'll investigate logs, pipeline state, and recent changes to help figure out what's happening.
```

When invoked WITHOUT parameters:
```
I'll help debug your current issue.

Please describe what's going wrong:
- What are you working on?
- What specific problem occurred?
- When did it last work?

I can investigate logs, test output, and recent changes to help identify the issue.
```

## Process Steps

### Step 1: Understand the Problem

After the user describes the issue:

1. **Read any provided context** (plan or ticket file):
   - Understand what they're implementing/testing
   - Note which phase or step they're on
   - Identify expected vs actual behavior

2. **Quick state check**:
   - Current git branch and recent commits
   - Any uncommitted changes
   - When the issue started occurring

### Step 2: Investigate the Issue

Spawn parallel Task agents for efficient investigation:

```
Task 1 - Check Recent Test Output:
Run the relevant tests and capture output:
1. poetry run pytest -x -v (stop at first failure, verbose)
2. Analyze the error messages and stack traces
3. Look for patterns in failures
Return: Key errors/warnings with context
```

```
Task 2 - Type and Lint Check:
Run static analysis:
1. poetry run pyright - check for type errors
2. poetry run ruff check - check for lint issues
3. Note any errors related to the problem area
Return: Static analysis findings
```

```
Task 3 - Git and File State:
Understand what changed recently:
1. Check git status and current branch
2. Look at recent commits: git log --oneline -10
3. Check uncommitted changes: git diff
4. Verify expected files exist
Return: Git state and any file issues
```

### Step 3: Present Findings

Based on the investigation, present a focused debug report:

```markdown
## Debug Report

### What's Wrong
[Clear statement of the issue based on evidence]

### Evidence Found

**From Tests**:
- [Error/failure with context]
- [Pattern or repeated issue]

**From Static Analysis**:
- [Type errors or lint issues]

**From Git/Files**:
- [Recent changes that might be related]
- [File state issues]

### Root Cause
[Most likely explanation based on evidence]

### Next Steps

1. **Try This First**:
   ```bash
   [Specific command or action]
   ```

2. **If That Doesn't Work**:
   - Check Docker container logs: `docker compose logs dagster-dev`
   - Run specific test: `poetry run pytest path/to/test.py -v`
   - Check Dagster UI for pipeline errors

### Can't Access?
Some issues might be outside my reach:
- Dagster UI pipeline state
- External service availability (BigQuery, GCS, NCBI)
- Docker container internal state

Would you like me to investigate something specific further?
```

## Important Notes

- **Focus on the problem** - don't go on tangents
- **Always require problem description** - can't debug without knowing what's wrong
- **Read files completely** - no limit/offset when reading context
- **Guide back to user** - some issues require manual investigation
- **No file editing** - pure investigation only

## Quick Reference

**Run Tests**:
```bash
poetry run pytest -x -v                                    # All tests, stop at first failure
poetry run pytest projects/edna_dagster_pipelines/tests -v  # Dagster tests
poetry run pytest libraries/core-analysis-lib/tests -v      # Core lib tests
```

**Static Analysis**:
```bash
poetry run pyright     # Type checking
poetry run ruff check  # Linting
```

**Docker/Dagster**:
```bash
docker compose logs dagster-dev          # Dagster logs
docker compose run dagster-dev bash      # Shell into container
```

**Git State**:
```bash
git status
git log --oneline -10
git diff
```
