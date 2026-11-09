# Sentinel Security Ledger

## 2026-11-09 - Audit of `internal/` State Management, Processors, and Boundary Handlers

**Vulnerability:** None (Confirmed clean perimeter).
**Learning:**
- `internal/configurable`: Agent and workflow references validate path containment using `filepath.Abs` and `filepath.EvalSymlinks`, checking `strings.HasPrefix(checkPath, parentDir + string(os.PathSeparator)) || checkPath == parentDir` to strictly prevent path traversal across agent directories and workflow DAG references.
- `internal/llminternal/request_confirmation_processor.go`: Implements fail-closed detection against ambiguous or tampered HITL tool confirmations across multiple session events (`conflictingConfirmations`), ensuring tool arguments cannot be substituted under legitimate approvals.
- `internal/originguard`: Strictly binds loopback requests and distinguishes DNS rebinding attacks from same-machine reverse proxies by inspecting bare hostnames, port stripping, and verifying loopback IPs via `netip.ParseAddr`.
- `internal/authhandler`: Redacts client secrets before writing to `temp:` or session state (`storedCredential`), pins client IDs from configurations, and uses bounded timeouts (`tokenRequestTimeout = 10s`) for external token endpoints.
- `internal/sessionutils`: Sanitizes session state prefixes (`app:`, `user:`, `temp:`) and enforces bidirectional json marshalling roundtrips (`CheckRequestedAuthConfigs`) to prevent state corruption before event persistence.

**Prevention:** Maintain zero-allocation fast paths on request-processing turns while preserving defensive validation at all external boundary deserializers.
