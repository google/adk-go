# qa_evaluator

Runs a real Gemini FAQ-bot agent against a small eval set and grades
every response with a Gemini-backed **LLM-as-judge** scorer.

## Run

```
export GOOGLE_API_KEY=...
go run ./examples/v2/qa_evaluator/
```

(Non-interactive — prints a JSON report and a pass-rate summary.)

## What it shows

- A `runner.Runner` driving an `llmagent.New(...)` Gemini agent.
- An `eval.Runner` calling that agent once per `eval.Case`.
- An `eval/llmjudge` scorer that, for each case, asks the SAME Gemini
  model to grade how well the agent's answer matches the reference
  (strict JSON `{score, reason}` response).
- A pass/fail rollup with a 0.7 threshold.

Wire your real agent (with whatever tools, plugins, sub-agents you
already use) into the `agentAdapter` to evaluate it offline against
your own eval set.
