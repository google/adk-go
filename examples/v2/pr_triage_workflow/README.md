# pr_triage_workflow

Automates the first-pass triage of a GitHub pull request as a workflow.

```
START → fetch_pr ─┬─ security_scan ─┐
                  ├─ check_breaking ─┼─ join → decide
                  └─ suggest_labels ─┘
```

What it shows:

- `workflow.Workflow` with fan-out from a single source into three
  parallel scan nodes, then a `JoinNode` aggregates their outputs,
  then a final decision node summarizes.
- `WithRetry` on `fetch_pr` simulates a flaky upstream API: the first
  call fails with a transient 503, the retry succeeds.
- `WithTimeout` on the security scan caps how long the slow scanner
  can block the pipeline.

The PR fetch is mocked via a deterministic in-memory database — replace
the `fetchPR` body with a real GitHub call (e.g. `gh api`) when wiring
this into a production triager.
