# Review Pull Request

Performs comprehensive automated code review of GitHub pull requests.

## Usage

**Review current branch PR:**
```bash
/review_pr
```

**Review specific PR:**
```bash
/review_pr
# Then select from list
```

## What This Command Does

1. Identifies the PR to review
2. Retrieves PR changes and metadata
3. Analyzes architectural context
4. Reviews code in batches (if large PR)
5. Aggregates findings and calculates compliance score
6. Generates detailed review document
7. Optionally posts review as PR comment

## Output

- **Review Document**: `thoughts/shared/pr_reviews/{number}_review.md`
- **Compliance Score**: 0-100 (70+ recommended for merge)
- **Categorized Findings**: MUST FIX / SHOULD FIX / CONSIDER
- **Recommendations**: Prioritized action items

## Tips

- **Large PRs**: Reviews 50+ files may take several minutes
- **Context**: Command analyzes existing patterns for consistency
- **Scoring**: Weighted by file count across batches
- **Iterations**: Re-run after fixes to verify improvements
- **Comments**: Can post full review or summary to PR

## Requirements

- GitHub CLI (`gh`) authenticated and configured
- Default repository set (run `gh repo set-default` if needed)
- PR must exist and be accessible

---

You are tasked with performing a comprehensive automated code review of a GitHub pull request following the repository's coding standards and best practices.

## Steps to follow:

### 1. Identify the PR to Review

**Check for current branch PR:**
- Run: `gh pr view --json url,number,title,state,author 2>/dev/null`
- If successful, use this PR

**If no PR on current branch:**
- Run: `gh pr list --limit 10 --json number,title,headRefName,author,createdAt --state open`
- Present the list to user
- Ask: "Which PR would you like me to review? (Enter PR number)"
- Wait for user input

**Validate selection:**
- Confirm PR number with user
- Run: `gh pr view {number} --json url,title,number,state,author`
- Display PR details and confirm

### 2. Gather PR Information

**Retrieve comprehensive PR data:**
```bash
# Get PR metadata
gh pr view {number} --json url,title,number,state,author,baseRefName,headRefName,createdAt,updatedAt

# Get full diff
gh pr diff {number}

# Get list of changed files
gh pr view {number} --json files

# Get commit history
gh pr view {number} --json commits
```

**Handle errors:**
- If "no default remote repository" error, instruct user to run `gh repo set-default`
- If PR not found, verify PR number and retry

### 3. Analyze Changed Files

**Parse the files JSON:**
- Extract list of changed file paths
- Count total files changed
- Filter for reviewable files (`.py`)
- Exclude generated files, config files (unless security-related)
- Categorize files by directory/feature

**Determine batching strategy:**
- If ≤15 files: Single batch
- If 16-50 files: 2-3 batches (group by feature/directory)
- If >50 files: Multiple batches (10-15 files each, grouped logically)

**Display batching plan:**
```
I found {N} reviewable files in this PR.
I'll review them in {M} batches:
- Batch 1: {count} files in {area}
- Batch 2: {count} files in {area}
...
```

**File Categorization:**

**Determine file priority:**
- P0 (Critical): Security, auth, database schema, API routes
- P1 (High): Core business logic, data access, validation
- P2 (Medium): Utilities, helpers, CLI interfaces
- P3 (Low): Tests, config, documentation

**Group files by feature:**
- Parse file paths for feature indicators
- Group files in same `features/` subdirectory
- Keep core function + op/asset + test in same batch

**Create batches:**
Use this batching logic:
- Group by feature/directory first
- If group fits in current batch (max 12 files), add it
- Otherwise start new batch with this group
- If single group exceeds batch size, split it
- Keep related files together (core function + op/asset + test)

**Batch metadata:**
- Batch number
- File count
- Feature/component area
- Priority level (based on file priorities)

**Exclude files that shouldn't be reviewed:**
- `__pycache__/`, `.pyc`, `venv/`, `.venv/`, `dist/`, `build/`
- `.lock` files, generated config files (unless security-relevant)
- Migration files (review separately if needed)

### 4. Conduct Context Analysis

**Spawn parallel context research agents:**

Before running code-reviewer, gather architectural context by spawning these agents in parallel:

**Agent 1 - codebase-analyzer:**
```
Analyze the architectural context for PR #{number}.

Files changed: {list of changed files}

Understand:
1. What features/components are being modified
2. How these components fit into the overall architecture
3. Key dependencies and integration points
4. Related code that wasn't changed but is relevant
5. Existing patterns in these areas

Return: Architectural overview with file:line references to relevant existing code
```

**Agent 2 - codebase-pattern-finder:**
```
Find existing patterns related to the changes in PR #{number}.

Files changed: {list of changed files}

Search for:
1. Similar implementations in the codebase
2. Established patterns for these types of changes
3. Test patterns for similar features
4. Error handling patterns in related code

Return: Concrete examples of existing patterns with file:line references
```

**Wait for both agents to complete before proceeding.**

**Generate context summary from agent results:**

Create a context summary in this format:
```markdown
## Architectural Context

### Components Modified
[From codebase-analyzer]
- Component: {name} at {file:line}
  - Role: {description}
  - Dependencies: {list}
  - Integration points: {description}

### Relevant Patterns
[From codebase-pattern-finder]
- Pattern: {name}
  - Found in: {file:line}
  - Used for: {description}
  - Key aspects: {list}
```

### 5. Run Batched Code Reviews

**Progress Tracking:**

Before starting reviews, create todos for tracking:
- Create a todo for each batch: "Review Batch {N}: {area} ({count} files)"
- Create todos for: "Aggregate review findings" and "Generate review document"
- Use TodoWrite tool to create these todos

For each batch of files:

**Display progress:**
```
Reviewing Batch {N} of {M}: {area} ({count} files)...
```

**Mark batch as in_progress in todo list**

**Invoke code-reviewer agent:**

Use the Task tool to spawn a code-reviewer agent with this prompt:

```
Review the following files from PR #{number} as Batch {N} of {M}:

{list of file paths}

Context from analysis:
{summary of architectural context from step 4}

Relevant patterns:
{summary of patterns from step 4}

These files are part of a larger PR changing {description}. Focus on:
1. Adherence to project rules (.claude/guidelines/rules.md)
2. Consistency with coding principles (.claude/guidelines/coding_principles.md)
3. Application of best practices (.claude/guidelines/best_practices.md)
4. Consistency with existing patterns you have access to

If you need additional context about how existing code works, use your Read tool to examine related files.
If you need to find similar patterns, use your Grep and Glob tools.

Return a detailed review report following your standard format.
```

**Wait for code-reviewer to complete**

**Display batch results:**
```
✓ Batch {N} complete: {total} issues found ({must_fix} MUST FIX, {should_fix} SHOULD FIX, {consider} CONSIDER)
```

**Mark batch as completed in todo list**

**Store batch results in memory**

**Continue for all batches.**

### 6. Aggregate Review Findings

**Mark aggregation todo as in_progress**

**Collect all batch results:**
- Compile findings from all code-reviewer invocations
- Merge issue lists by category (MUST FIX, SHOULD FIX, CONSIDER)
- Remove duplicate findings (same file:line)
- Sort findings by severity and file path

**Deduplication logic:**
- Remove duplicates based on file path and category
- Keep first occurrence of each unique finding
- Preserve all details from original finding

**Sort findings:**
- Within each severity level, group by category (Types & Safety, Errors, etc.)
- Sort by file path alphabetically
- Sort by line number within file

**Calculate overall compliance score:**

For each batch:
- batch_issues = MUST FIX count, SHOULD FIX count, CONSIDER count
- batch_score = 100 - (MUST FIX × 10 + SHOULD FIX × 5 + CONSIDER × 2)
- batch_weight = file count in batch / total files

Overall Score = Weighted average of batch scores
Overall Interpretation = Based on overall score:
- 90-100: Excellent - Production ready
- 70-89: Good - Minor issues to address
- 50-69: Fair - Significant improvements needed
- <50: Poor - Major violations, do not merge

**Determine recommendation:**
- Score ≥90 AND no MUST FIX: "APPROVE" (with minor comments)
- Score 70-89 AND no MUST FIX: "COMMENT" (issues to consider)
- Score 50-69 OR has MUST FIX: "REQUEST CHANGES" (significant issues)
- Score <50: "REQUEST CHANGES" (critical issues)

**Generate category compliance table:**

For each rule category (Types & Safety, DataFrames, Runtime Validation, Errors, Logging, Dagster, Testing, Security, Performance):
- Count issues in category across all findings
- Determine status:
  - ❌ if any MUST FIX in category
  - ⚠️ if 3+ SHOULD FIX or total issues > 3
  - ✅ if 0 issues
- Generate note describing the issues

**Extract architectural summary:**
- Use context from step 4 (analyzer and pattern-finder results)
- Extract architectural observations from code review findings
- Identify new patterns introduced
- Note deviations from existing patterns
- Parse diff for package.json changes (new dependencies)
- Combine into summary

**Generate testing summary:**
- Filter changed files for test files (`test_*.py`, `*_test.py`, `tests/`)
- Count test files added vs modified
- For each non-test file, check if corresponding test file changed
- List files without test coverage
- Extract test-related findings from code review

**Mark aggregation todo as completed**

### 7. Generate Review Document

**Mark document generation todo as in_progress**

**Read the PR review template:**
- File: `thoughts/shared/pr_review_template.md`
- If template doesn't exist, inform user and use fallback structure

**Fill template sections:**

**Executive Summary:**
- Overall compliance score and interpretation
- Total issue counts by severity
- Recommendation (APPROVE / COMMENT / REQUEST CHANGES)

**Context Analysis:**
- Brief summary of what the PR does (from diff analysis)
- Files changed count and complexity assessment
- Risk level based on areas modified

**Detailed Findings:**
- For each finding from aggregated results:
  - Category and issue title
  - File path with line number
  - Rule violated with reference
  - Detailed description
  - Why it matters
  - Recommended fix with code example

**Compliance by Category:**
- Fill table with status for each category
- Use batch results to determine status

**Files Reviewed:**
- List all files organized by batch
- Include issue counts and visual indicators

**Architectural Observations:**
- Use context from step 4
- Note patterns used
- Consistency with codebase
- New dependencies

**Testing Assessment:**
- Identify test files in changed files
- Note coverage additions/modifications
- Highlight missing coverage

**Security & Performance Notes:**
- Extract security and performance findings
- Highlight any concerns

**Recommendations for Author:**
- Prioritize action items:
  - Priority 1: All MUST FIX items
  - Priority 2: Critical SHOULD FIX items
  - Priority 3: CONSIDER items and minor SHOULD FIX

**Review Batches Summary:**
- List each batch with file count and score
- Show weighted average calculation

**Verification Commands:**
- Include standard verification commands

**Mark document generation todo as completed**

### 8. Save Review Document

**Ensure directory exists:**
```bash
mkdir -p thoughts/shared/pr_reviews/
```

**Determine file path:**
```
thoughts/shared/pr_reviews/{pr_number}_review.md
```

**Check for existing review:**
- If file exists, read it first
- Inform user this is an updated review
- Archive old review with timestamp:
  - Archive path: `thoughts/shared/pr_reviews/{pr_number}_review_{timestamp}.md`
  - Timestamp format: YYYY-MM-DD_HH-MM-SS

**Write new review:**
- Use Write tool to save filled template
- Confirm save was successful
- Show file path to user

**Display summary:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Review Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PR: #{pr_number} - {pr_title}
Overall Score: {score}/100 - {interpretation}

Issues Found:
  🔴 MUST FIX:    {count} critical issues
  🟡 SHOULD FIX:  {count} standard violations
  🔵 CONSIDER:    {count} improvement suggestions

Recommendation: {recommendation}

Review saved to:
  {review_file_path}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 9. Prompt for PR Comment Posting

**Display options:**
```
Would you like to post this review as a comment on PR #{number}?

Options:
  1. Yes - Post full review as comment
  2. Yes - Post summary only (with link to full review)
  3. No - I'll review the file first

Enter your choice (1/2/3):
```

**Wait for user input:**
- Parse user response
- Handle variations: "yes", "y", "1", "full", "summary", "2", "no", "n", "3"
- Re-prompt if unclear

### 10. Post Review as PR Comment (If Approved)

**If user chooses option 1 (full review):**

**Prepare full review comment:**
```markdown
# 🤖 Automated Code Review

**Generated**: {timestamp}
**Overall Score**: {score}/100 - {interpretation}

{entire review content}

---

*This review was generated by Claude Code. Review document: `thoughts/shared/pr_reviews/{number}_review.md`*

*To verify fixes, run:*
```bash
poetry run ruff format && poetry run ruff check && poetry run pyright && poetry run pytest
```
```

**Post full comment:**
```bash
gh pr comment {number} --body "$(cat <<'EOF'
{full_review_comment}
EOF
)"
```

**If user chooses option 2 (summary only):**

**Prepare summary comment:**
```markdown
# 🤖 Automated Code Review

**Generated**: {timestamp}
**Overall Score**: {score}/100 - {interpretation}
**Recommendation**: {recommendation}

## Summary
- 🔴 **MUST FIX**: {count} critical issues
- 🟡 **SHOULD FIX**: {count} standard violations
- 🔵 **CONSIDER**: {count} improvement suggestions

## Top Priority Issues

{First 5 MUST FIX findings - condensed format}

## Key Recommendations

### Priority 1 (Before Merge)
{Top 3-5 action items}

### Priority 2 (Consider Before Merge)
{Top 3-5 action items}

---

**📄 Full detailed review**: `thoughts/shared/pr_reviews/{number}_review.md`

**✓ Verification commands**:
```bash
poetry run ruff format && poetry run ruff check && poetry run pyright && poetry run pytest
```

---

*Automated review by Claude Code*
```

**Post summary comment:**
```bash
gh pr comment {number} --body "$(cat <<'EOF'
{summary_comment}
EOF
)"
```

**Handle posting success:**
```bash
if [ $? -eq 0 ]; then
    echo "✓ Review comment posted successfully"

    # Get PR URL
    pr_url=$(gh pr view {number} --json url -q .url)
    echo "View PR: $pr_url"
else
    echo "✗ Failed to post comment"
    echo "You can post manually with:"
    echo "  gh pr comment {number} --body-file thoughts/shared/pr_reviews/{number}_review.md"
fi
```

**If user chooses option 3 (no):**

Display:
```
Review saved to: thoughts/shared/pr_reviews/{number}_review.md

You can post it later with:
  gh pr comment {number} --body-file thoughts/shared/pr_reviews/{number}_review.md

Or post just the summary by editing the file first.
```

## Important Notes

### Context Analysis Strategy
- Run context analysis ONCE before all batches
- Share context summary with each batch review
- This prevents redundant analysis and ensures consistency
- Allow code-reviewer to call other agents if it needs more info

### Score Aggregation
- Weight batch scores by file count
- Round final score to nearest integer
- Include both weighted average and per-batch scores in report
- Interpretation based on overall score, not batch scores

### Error Handling
- If code-reviewer fails on a batch, continue with other batches
- Note failed batch in final report
- Provide error details to user
- Allow retry of individual batches

### Performance Considerations
- For very large PRs (100+ files), warn user about review time
- Consider reviewing in multiple sessions
- Allow user to specify subset of files to review
- Cache context analysis results
