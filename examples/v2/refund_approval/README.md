# refund_approval

Refund processing with a manager-approval gate. Demonstrates
**human-in-the-loop** (`workflow.RequestInput`) plus
**resume-from-session-events** (`Runner.Resume`).

```
START → validate ─→ check_amount ─→ wait_for_manager ─→ process_refund
                          │             (HITL pause)
                          └─→ auto_approve (if amount < $50)
```

Two scenarios run back-to-back:

1. **Small refund ($23.50)** — auto-approves, processes immediately.
2. **Large refund ($420.00)** — pauses on `wait_for_manager` with a
   `RequestInput` interrupt. The demo then "resumes" by sending a
   `FunctionResponse` containing the manager's decision; the workflow
   re-enters the same session, reads the response via
   `ctx.ResumeInput`, and processes the refund.

The pause/resume pattern is durable — the session events carry enough
information to replay the workflow from the interrupt point even after
the runner restarts.
