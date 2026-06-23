# Agent Development Kit (ADK) for Go v 2.0

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)


## Breaking changes

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

var _ agent.CallbackContext = (*MockCallbackContext)(nil)
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

Assert against the unified `agent.Context` directly. `CallbackContext` and
`ToolContext` are only transitional aliases during the migration — don't rely on
them; migrate straight to `agent.Context`.

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
