## Updates Made

@verdverm I've addressed all the issues you mentioned:

### 1. ✅ Merge Conflict Resolved
Merged `upstream/main` into the branch successfully. All conflicts are now resolved.

### 2. ✅ Fixed Example Imports and API Usage
The example now uses the correct, current API:
- Changed from `google.golang.org/adk/llm` → `google.golang.org/adk/agent/llmagent`
- Changed from `google.golang.org/adk/runner/full` → `google.golang.org/adk/cmd/launcher/full`
- Updated to use `llmagent.New()` instead of deprecated `agent.NewLLMAgent()`
- Updated to use `launcher.Config` with the new API
- Added proper Gemini model initialization with API key

The example should now build and run correctly with `go mod tidy` and `go run main.go`.

### 3. ✅ Removed .DS_Store
- Removed `.DS_Store` from the repository
- Added it to `.gitignore` to prevent future commits

### Regarding Your Questions

**Per-session confirmation overrides**: That's a great feature idea! The current implementation provides the foundation for this. To implement "don't ask again for this session", we could:
- Add a session-level confirmation cache
- Store approved tool+payload combinations
- Check the cache before requesting confirmation

This could be a follow-up enhancement. Would you like me to open a separate issue for tracking this feature?

**Return payload from confirmation**: You're absolutely right that this is more powerful than just yes/no. The current `ConfirmationRequest` includes an arbitrary `Payload` field that could be used for:
- Presenting options (e.g., flight choices)
- Showing diffs for file writes
- Allowing partial selections

The `ConfirmationResponse` could be extended to include user selections/modifications. This would make it a general-purpose "user interrupt" mechanism rather than just confirmation.

Should we expand the scope of this PR to include response payloads, or handle that in a follow-up?

All changes have been pushed. The PR should now be ready for final review.
