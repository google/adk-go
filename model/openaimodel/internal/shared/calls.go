// Copyright 2025 Google LLC
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

package shared

import (
	"encoding/json"
	"fmt"

	"google.golang.org/genai"
)

// CallTracker helps us manage function call IDs, ensuring that function responses
// can be correctly associated with their corresponding calls, especially when IDs are not
// explicitly provided in the input.
type CallTracker struct {
	NextID  int
	Pending []string
}

// ResolveResponseID reports the call ID a function response answers, consuming
// it from the pending list. An unset ID pairs with the oldest outstanding call,
// which is the only pairing available when the caller did not supply one.
func (t *CallTracker) ResolveResponseID(fr *genai.FunctionResponse) (string, error) {
	if fr.ID == "" {
		if len(t.Pending) == 0 {
			return "", fmt.Errorf("openai: response for %q missing call id", fr.Name)
		}
		callID := t.Pending[0]
		t.Pending = t.Pending[1:]
		return callID, nil
	}
	for i, pending := range t.Pending {
		if pending == fr.ID {
			t.Pending = append(t.Pending[:i], t.Pending[i+1:]...)
			return fr.ID, nil
		}
	}
	return "", fmt.Errorf("openai: received function response for unknown or already completed call id %q", fr.ID)
}

// TakeCallID reports the ID to send for a function call, minting one when the
// caller left it unset so the matching response can still be paired.
func (t *CallTracker) TakeCallID(fc *genai.FunctionCall) string {
	callID := fc.ID
	if callID == "" {
		callID = fmt.Sprintf("adk-openai-call-%d", t.NextID)
		t.NextID++
	}
	t.Pending = append(t.Pending, callID)
	return callID
}

// MarshalFunctionArgs encodes a call's arguments, reading a nil map as a call
// that takes none rather than as JSON null.
func MarshalFunctionArgs(args map[string]any) (string, error) {
	if args == nil {
		args = map[string]any{}
	}
	encoded, err := json.Marshal(args)
	if err != nil {
		return "", fmt.Errorf("openai: marshal function args: %w", err)
	}
	return string(encoded), nil
}
