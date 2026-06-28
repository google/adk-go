# Agent Development Kit (ADK) for Go v 2.0

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)


## Breaking changes

### `session.NewEvent` now requires a `context.Context`

`session.NewEvent` takes a `context.Context` as its first argument:

```go
func NewEvent(ctx context.Context, invocationID string) *Event
```

The event ID and timestamp are now obtained through the `platform` package, so a
time or UUID provider installed on `ctx` (see `platform.WithTimeProvider` and
`platform.WithUUIDProvider`) controls them. This lets callers such as workflow
engines produce deterministic, replay-safe events.

The previous parameterless-context form and the temporary `NewEventWithContext`
helper are gone. Migrate by passing the context that is already in scope:

```go
// Before
ev := session.NewEvent(ctx.InvocationID())
// or
ev := session.NewEventWithContext(ctx, ctx.InvocationID())

// After
ev := session.NewEvent(ctx, ctx.InvocationID())
```

Any `context.Context` works as the first argument: the `ctx` of an agent/tool/
callback (which embed `context.Context`), the incoming RPC/HTTP request context,
or — in tests — `t.Context()`. Per
[go/how-to-use-a-context](http://go/how-to-use-a-context), thread the context
down the call chain rather than creating one with `context.Background()` in the
middle of it; reserve `context.Background()` for `main`, `init`, and top-level
test/setup code.

If you call `NewEvent` from a helper that does not yet receive a context, add a
`ctx context.Context` parameter to that helper and pass it through from its
callers.

### Mocks update required for unified contexts 
PR https://github.com/google/adk-go/pull/945 merges ToolContext and CallbackContext into single Context. 
ToolContext and CallbackContext became aliases to Context. 

This introduces a problem for mock contexts - new functions (ToolContext-related) are missing if the mock was created for the previous version of CallbackContext. 
Solution:
Add those functions to your mock:
```go
func (m *MockCallbackContext) Actions() *session.EventActions                       { return nil }
func (m *MockCallbackContext) FunctionCallID() string                               { return "" }
func (m *MockCallbackContext) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }
func (m *MockCallbackContext) RequestConfirmation(hint string, payload any) error {
	return fmt.Errorf("RequestConfirmation() is not supported for MockCallbackContext")
}
func (m *MockCallbackContext) SearchMemory(ctx context.Context, query string) (*memory.SearchResponse, error) {
	return nil, fmt.Errorf("SearchMemory() is not supported for MockCallbackContext")
}

var _ agent.Context = (*MockCallbackContext)(nil)
```

#### Alternative: embed `agent.StrictContextMock`

Adding each missing method by hand is reactive: every time the context surface
grows, your mocks break again and you have to patch them. Instead, you can embed
`agent.StrictContextMock` in your test fake and override only the methods your
test actually uses. Because it implements the whole unified context surface,
embedders keep compiling as the interface grows — no further edits needed when
methods are added.

Un-overridden methods panic with "not implemented", so an unexpected call fails
the test loudly instead of silently returning a zero value. The standard
`context.Context` methods (`Deadline`, `Done`, `Err`, `Value`) read from the
supplied `Ctx` rather than panicking.

Assert against the unified `agent.Context` directly. The transitional
`CallbackContext` and `ToolContext` aliases have been removed — migrate any
remaining references straight to `agent.Context`.

```go
// Embed StrictContextMock and override only what the test needs.
type fakeContext struct {
	agent.StrictContextMock
}

var _ agent.Context = (*fakeContext)(nil)

func TestSomething(t *testing.T) {
	cc := &fakeContext{agent.StrictContextMock{Ctx: context.Background()}}
	// Override methods as needed, e.g. by adding them on fakeContext.
	// ...
}
```
