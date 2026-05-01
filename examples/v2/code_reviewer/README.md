# code_reviewer

Real Gemini code-review agent. Receives a Python snippet, runs it via
`codeexec/unsafelocal`, and writes a short review of the result.

## Run

```
export GOOGLE_API_KEY=...
go run ./examples/v2/code_reviewer/             # console
go run ./examples/v2/code_reviewer/ web         # adk-web
```

## Try it

> Run this Python and review:
> ```python
> def add(a, b): return a + b
> assert add(2, 3) == 5
> print("PASS")
> ```

Agent calls `run_python` → `{stdout: "PASS\n", exit_code: 0, ...}` →
writes a "passed" review.

> Now try:
> ```python
> def divide(a, b): return a // b
> assert divide(7, 2) == 3.5
> ```

Agent runs it → AssertionError → review points out the integer-division
bug.

## SECURITY

`unsafelocal` provides **no sandbox**. The subprocess runs with the
agent's full privileges. Do not point this at attacker-controlled
input. For production replace with `codeexec/container`,
`codeexec/vertexai`, `codeexec/gke`, or `codeexec/agentenginesandbox`.
