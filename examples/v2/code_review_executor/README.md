# code_review_executor

Code-review agent that runs the reviewer's Python snippet in a
subprocess sandbox and reports stdout / stderr / exit code. Demonstrates
the `codeexec` interface with the `unsafelocal` backend.

The pattern: a reviewer drops a code snippet into a PR comment with a
small test harness, and a bot agent executes it locally to surface the
output. In production replace `unsafelocal` with `container`,
`vertexai`, `gke`, or `agentenginesandbox` per your isolation needs.

The example runs three snippets:
1. A correct fix (test passes).
2. A broken fix (assertion fails).
3. A snippet that exceeds the runtime budget (gets killed by `MaxRuntime`).

Skips automatically when no Python interpreter is on PATH.
