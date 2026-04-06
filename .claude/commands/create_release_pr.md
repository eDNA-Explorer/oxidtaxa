# Create a Release PR from develop to main

You are tasked with creating a release pull request that merges the develop branch into main, with a comprehensive summary of all changes included since the last release.

## Steps to follow:

1. **Read the release PR description template:**
   - First, check if `thoughts/shared/release_pr_description.md` exists
   - If it doesn't exist, inform the user that they need to create a release PR description template at `thoughts/shared/release_pr_description.md`
   - Read the template carefully to understand all sections and requirements

2. **Verify branch status:**
   - Ensure we're on the develop branch or switch to it: `git checkout develop`
   - Pull latest changes: `git fetch origin && git pull origin develop`
   - Verify main branch exists: `git rev-parse --verify origin/main`

3. **Find the last release point:**
   - Find the last merge from develop to main: `git log origin/main --first-parent --merges --grep="Merge.*develop" -1 --format=%H`
   - If no previous merge found, use the first commit in main as the baseline
   - Store this commit hash as the baseline for comparison

4. **Get all PRs merged since last release:**
   - Get all merge commits in develop since baseline: `git log {baseline}..origin/develop --merges --first-parent --format=%H`
   - For each merge commit:
     - Extract PR number from commit message (format: "Merge pull request #123")
     - Get PR details: `gh pr view {number} --json number,title,url,author,labels`
     - Store PR information for categorization

5. **Categorize PRs:**
   - Group PRs by label or title keywords:
     - **Features**: Labels containing "feature", "enhancement", or titles starting with "feat:"
     - **Bug Fixes**: Labels containing "bug", "fix" or titles starting with "fix:"
     - **Improvements**: Labels containing "improvement", "refactor" or titles starting with "refactor:", "perf:"
     - **Documentation**: Labels containing "documentation", "docs" or titles starting with "docs:"
     - **Other Changes**: Everything else
   - Identify any PRs with "breaking change" labels or keywords

6. **Generate version or date identifier:**
   - Ask the user if they want to use a version number (e.g., "v1.2.0") or today's date (e.g., "2025-10-10")
   - Use their preference as the release identifier

7. **Create the PR:**
   - Push develop branch if needed: `git push origin develop`
   - Create the PR: `gh pr create --base main --head develop --title "Release: {version/date}" --body ""`
   - Capture the PR number from the output

8. **Build the release description:**
   - Start with the template from `thoughts/shared/release_pr_description.md`
   - Fill in the release identifier
   - For each category, add PRs in this format:
     ```markdown
     - #{number} {title} by @{author} - [Link]({url})
     ```
   - Highlight any breaking changes prominently
   - Add notes about:
     - Any database migrations required
     - Environment variable changes
     - Deployment considerations
   - Review the checklist and mark any items you can verify automatically

9. **Save and update the PR:**
   - Write the completed description to `thoughts/shared/prs/{number}_release_description.md`
   - Update the PR description: `gh pr edit {number} --body-file thoughts/shared/prs/{number}_release_description.md`

10. **Present summary:**
    - Show the user:
      - Total number of PRs included
      - Breakdown by category
      - Any breaking changes or migration notes
      - PR URL
      - Items in the deployment checklist that need review

## Important notes:
- This command always uses `develop` as the source branch and `main` as the target
- The PR description should be a high-level summary, not implementation details
- Focus on user-facing changes and deployment considerations
- Group related PRs together when possible
- Highlight any risks or important considerations
- If there are no changes between develop and main, inform the user and don't create a PR
- Always verify that the develop branch is ahead of main before creating the PR

## Error handling:
- If you get an error about no default remote repository, instruct the user to run `gh repo set-default`
- If develop and main are identical, inform the user no release is needed
- If there are uncommitted changes, ask the user to commit or stash them first
- If git commands fail, provide clear error messages and suggested fixes
