# incident_responder

Real Gemini-backed on-call agent. Holds a library of runbook **skills**
in a registry; the model loads only the runbook(s) relevant to the
current incident, keeping the system prompt lean.

## Run

```
export GOOGLE_API_KEY=...
go run ./examples/v2/incident_responder/             # console
go run ./examples/v2/incident_responder/ web         # adk-web
```

## Try it

> Page received: PD-90234, type=db_replication_lag. Standby lag is 412
> seconds, threshold is 60. Triage and decide next steps.

Watch the agent:

1. Call `list_skills(query="replication")` — registry returns
   `replication-runbook`.
2. Call `load_skill(name="replication-runbook")` — the runbook's
   instructions become part of the system prompt going forward (via
   `SkillsInstructionPlugin`); the body is also returned in the
   `load_skill` response so the model has it on the next turn.
3. Follow the runbook: compute lag, check network, decide whether to
   fail the standby out of read traffic.

The other 3 runbooks (`database-runbook`, `k8s-runbook`,
`network-runbook`) stay dormant — the model never has to filter them
out of its context.
