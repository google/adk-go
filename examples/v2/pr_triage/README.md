# pr_triage

Real Gemini PR-triage agent. Coordinator with three scan tools
(security, breaking-change, labeling). The model orchestrates them and
produces a single triage decision.

## Run

```
export GOOGLE_API_KEY=...
go run ./examples/v2/pr_triage/             # console
go run ./examples/v2/pr_triage/ web         # adk-web
```

## Try it

> Triage PR #1234. Title: "Add OAuth2 token refresh to auth
> middleware". Files: auth/middleware.go, auth/oauth.go,
> auth/oauth_test.go. Description: implements automatic token refresh
> when upstream returns 401. Lines changed: 240.

Watch the model:

1. Call `security_scan` with the PR fields → returns `medium` risk
   because of the auth path.
2. Call `check_breaking` → returns `false` (no public API touched).
3. Call `suggest_labels` → returns `["area/auth", "tests", "size/M"]`.
4. Compose: action = `request-changes`, reason = security review
   recommended, labels listed.

Replace each tool stub in `main.go` with a real GitHub fetcher / SAST /
labeler when wiring this into a real bot.
