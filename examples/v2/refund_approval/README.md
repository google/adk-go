# refund_approval

Real Gemini agent that processes refunds. Small refunds auto-approve;
large refunds open a ticket and require a manager to call
`approve_refund` (typically a separate manager session). Demonstrates
how the agent partitions a workflow that combines automation with
human-in-the-loop.

## Run

```
export GOOGLE_API_KEY=...
go run ./examples/v2/refund_approval/             # console
go run ./examples/v2/refund_approval/ web         # adk-web
```

## Try it

> Customer requests refund of $23.50 on order ORD-101. Reason:
> duplicate charge.

Agent calls `request_refund` → returns `auto_approved` → agent confirms
to the user.

> Customer requests refund of $420 on order ORD-202. Reason: lost
> shipment.

Agent calls `request_refund` → returns `needs_manager_approval` with a
ticket id → agent tells the user that a manager is reviewing.

Then a manager (in a separate session, or the same one for the demo)
says:

> Approve ticket RFD-ORD-202. Note: carrier confirmed lost; reissue.

Agent calls `approve_refund(ticket="RFD-ORD-202", approved=true,
notes=...)` and confirms.

For a workflow-driven version that uses
`workflow.RequestInput` + `Runner.Resume` to pause execution literally
(rather than tracking pending state externally), see the
`pr_triage` example which exercises the workflow engine directly.
