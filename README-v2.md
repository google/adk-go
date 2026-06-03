# Agent Development Kit (ADK) for Go v 2.0

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)


## Breaking changes

### Unified contexts
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
