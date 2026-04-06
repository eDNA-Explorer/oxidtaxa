# Implement Tests

You are tasked with writing tests based on an approved test plan. Your job is to turn that plan into working tests.

**Input**: $ARGUMENTS (path to a test plan document). If no path is provided, ask the user for one before proceeding.

## Step 1: Read the test plan

Read the file at the provided path. Extract:
- The test groups with their approved approaches (unit test, integration script, crafted-state test)
- The mock targets with their exact patch strings
- The implementation order
- The scope constraints (what is and isn't being tested)
- Any shared test utilities specified

## Step 2: Follow the approved test approaches

The test plan specifies the approach for each test group. Follow those decisions — do not second-guess them.

### Where tests go:
- **`tests/` directories** (CI/CD): Only pure unit tests with zero external dependencies. These run in `poetry run pytest` and must pass without credentials, network, or cloud access.
- **`scripts/` directory** (manual): Any test that calls real external APIs (GCS, GEE, DB), requires credentials, or touches real infrastructure. These are run manually with `poetry run python scripts/test_*.py` or `bash scripts/test_*.sh`. They are NOT part of the CI/CD pipeline.

### Testing philosophy: real data, surgical mocks

Do NOT extensively mock external APIs. The goal is to test with real infrastructure and real data wherever possible. Only mock or modify data at the **specific point** where you need to create the edge case condition.

Approaches for creating edge cases without heavy mocking:
- **Corrupt real data**: Fetch real data from the API, then modify the response to create the edge case (e.g., remove a field, zero out a counter, set a contradictory status)
- **Craft bad state in real storage**: Write a deliberately broken manifest/config to staging GCS, then run the real processing code against it
- **Mock only the injection point**: For example, if you need a GEE task to return UNKNOWN status, mock only `ee.data.getTaskStatus` and let everything else (GCS reads, manifest parsing, state sync) run for real
- **Use real but controlled inputs**: Use a real project/dataset but with known sample data that exercises the edge case naturally

Avoid:
- Mocking entire chains of calls (e.g., mocking GCS client + blob + download + parse)
- Returning bare primitives from mocks when the real code returns dataclasses or complex objects
- Tests where more code is mocked than actually runs — these prove nothing

### Mocking rules (when a mock is unavoidable):
- Use the **exact patch targets** from the research findings where available
- Patch top-level imports at the **consuming module's namespace** (`my_module.download_file`)
- Patch lazy/local imports at the **source module** (`storage.download_file`)
- Mock the **single function** at the injection point, not the whole dependency chain
- Return realistic types that match the real return value
- Verify the mock is being hit — a test that passes without calling the real code is worthless

## Step 3: Write the tests

Implement test groups in the order specified by the test plan.

Structure every test as: **Setup → Act → Assert → Cleanup**

### Happy path tests:
- Call the function with representative normal inputs
- Assert the return value matches expected output
- Assert any side effects happened (state written, external call made)
- These are your baseline — if a happy path test fails, the feature is broken

### Edge case tests:
- **Name every assertion** so failures are immediately diagnosable: `"Manifest stays PROCESSING when cancelled exports remain"` not just `assert status == "PROCESSING"`
- **Assert the negative**, not just the positive — verify the system did NOT falsely complete, did NOT silently drop errors, did NOT invent failures
- Each assertion maps to **one specific behavior** you're verifying

### Isolation rules (all tests):
- No shared state between tests — each creates its own data and cleans up after
- No ordering dependencies — tests pass in any order
- Unique identifiers per test run — use `test-{name}-{random}` patterns
- Cleanup runs unconditionally, even when assertions fail

## Step 4: Run and iterate

Run all tests. For each failure:

1. **Determine if the test is wrong or the code is wrong.** If the code behaves correctly but the expectation was wrong, fix the test. If the code has a bug, fix the code.

2. **Understand WHY before changing anything.** Common surprises:
   - Code has a save-on-change guard that doesn't trigger when expected
   - A status field is updated in memory but not persisted to storage
   - A mock wasn't applied because the import path was wrong
   - The crafted state was valid enough that the code handled it correctly (not actually an edge case)

3. **Re-run the full suite after every fix**, not just the failing test.

## Step 5: Document in the PR

Add a testing section to the PR description with:
1. Reference to the test plan: `{path}`
2. Total tests, total assertions, pass/fail counts
3. A table showing each test group, what it covers, and the scenarios tested
4. Exact commands to run the tests
5. Any edge cases identified but not tested, with a one-line reason why
