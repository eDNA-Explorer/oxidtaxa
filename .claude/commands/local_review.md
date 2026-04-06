# Local Review

You are tasked with setting up a local review environment for a colleague's branch.

## Process

When invoked with a parameter like `gh_username:branchName`:

1. **Parse the input**:
   - Extract GitHub username and branch name from the format `username:branchname`
   - If no parameter provided, ask for it in the format: `gh_username:branchName`

2. **Extract ticket information**:
   - Look for ticket numbers in the branch name (e.g., `eng-1696`, `ENG-1696`)
   - Use this to create a short worktree directory name
   - If no ticket found, use a sanitized version of the branch name

3. **Set up the remote and worktree**:
   - Check if the remote already exists using `git remote -v`
   - If not, add it: `git remote add USERNAME git@github.com:USERNAME/edna-explorer-data-pipelines-species-modelling`
   - Fetch from the remote: `git fetch USERNAME`
   - Create worktree: `git worktree add -b BRANCHNAME ~/wt/edna/SHORT_NAME USERNAME/BRANCHNAME`

4. **Configure the worktree**:
   - Copy Claude settings: `cp .claude/settings.local.json WORKTREE/.claude/`
   - Install dependencies: `cd WORKTREE && poetry install`

## Error Handling

- If worktree already exists, inform the user they need to remove it first
- If remote fetch fails, check if the username/repo exists
- If setup fails, provide the error but continue

## Example Usage

```
/local_review colleague:feature/improve-tronko-processing
```

This will:
- Add 'colleague' as a remote
- Create worktree at `~/wt/edna/improve-tronko-processing`
- Set up the environment with poetry install
