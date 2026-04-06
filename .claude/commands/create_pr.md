# Open or Update a PR

You are tasked with creating or updating a pull request with a comprehensive description following the repository's standard template.

## Steps to follow:

1. **Read the PR description template:**
   - First, check if `.github/pull_request_template.md` exists
   - If not, check if `thoughts/shared/pr_description.md` exists
   - If neither exists, inform the user and use a sensible default structure
   - Read the template carefully to understand all sections and requirements

2. **Check if a PR already exists for this branch:**
   - Run: `gh pr view --json url,number,title,state 2>/dev/null`
   - If a PR exists, skip to step 4 (no need to push or create)
   - If no PR exists, continue to step 3

3. **Push the current branch and create the PR:**
   - Get the current branch name: `git rev-parse --abbrev-ref HEAD`
   - Determine the base branch:
     - If the user provided one, use that
     - Otherwise default to `develop`
   - Push the branch to origin: `git push -u origin {current_branch}`
   - Create the PR: `gh pr create --base {base_branch} --head {current_branch} --title "{title}" --body ""`
   - Capture the PR number from the output

4. **Gather comprehensive PR information:**
   - Get the full PR diff: `gh pr diff {number}`
   - If you get an error about no default remote repository, instruct the user to run `gh repo set-default`
   - Get commit history: `gh pr view {number} --json commits`
   - Get PR metadata: `gh pr view {number} --json url,title,number,state,baseRefName`
   - Check for existing description: check if `thoughts/shared/prs/{number}_description.md` already exists

5. **Analyze the changes thoroughly:** (ultrathink about the code changes, their architectural implications, and potential impacts)
   - Read through the entire diff carefully
   - For context, read any files that are referenced but not shown in the diff
   - Understand the purpose and impact of each change
   - Identify user-facing changes vs internal implementation details
   - Look for breaking changes or migration requirements

6. **Handle verification requirements:**
   - Look for any checklist items in the template
   - For each verification step:
     - If it's a command you can run (`poetry run pytest`, `poetry run ruff check`, etc.), run it
     - If it passes, mark the checkbox as checked: `- [x]`
     - If it fails, keep it unchecked and note what failed: `- [ ]` with explanation
     - If it requires manual testing, leave unchecked and note for user
   - Document any verification steps you couldn't complete

7. **Generate the description:**
   - Fill out each section from the template thoroughly:
     - Answer each question/section based on your analysis
     - Be specific about problems solved and changes made
     - Focus on user impact where relevant
     - Include technical details in appropriate sections
     - Write a concise changelog entry
   - Ensure all checklist items are addressed (checked or explained)

8. **Save and update the PR:**
   - Write the completed description to `thoughts/shared/prs/{number}_description.md`
   - Update the PR description: `gh pr edit {number} --body-file thoughts/shared/prs/{number}_description.md`
   - Show the user the generated description and PR URL
   - If any verification steps remain unchecked, remind the user to complete them before merging

## Important notes:
- This command works for both creating new PRs and updating descriptions on existing PRs
- Be thorough but concise - descriptions should be scannable
- Focus on the "why" as much as the "what"
- Include any breaking changes or migration notes prominently
- If the PR touches multiple components, organize the description accordingly
- Always attempt to run verification commands when possible
- Clearly communicate which verification steps need manual testing
