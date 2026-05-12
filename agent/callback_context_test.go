// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package agent

import (
	"testing"
)

// TestCallbackContext_OutOfToolCallReturnsZero verifies the
// runtime-check pattern for tool-call-only methods on a CallbackContext
// (which is now an alias of Context).
func TestCallbackContext_OutOfToolCallReturnsZero(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{})
	callback := NewCallbackContext(inv)

	if got := callback.FunctionCallID(); got != "" {
		t.Errorf("FunctionCallID() = %q, want empty (callback is not a tool call)", got)
	}
	if got := callback.Actions(); got != nil {
		t.Errorf("Actions() = %v, want nil (callback is not a tool call)", got)
	}
	if got := callback.ToolConfirmation(); got != nil {
		t.Errorf("ToolConfirmation() = %v, want nil (callback is not a tool call)", got)
	}
	if got := callback.RequestConfirmation("hint", nil); got != ErrOutsideToolCall {
		t.Errorf("RequestConfirmation() error = %v, want ErrOutsideToolCall", got)
	}
}
