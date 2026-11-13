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

package redis

import (
	"testing"

	"google.golang.org/adk/v2/model"
)

// An event with Partial true keeps it through the session document, as
// session.Event's own encoding keeps it. AppendEvent stores no partial event,
// so only the document codec can show it.
func TestSessionDocument_PartialTrueEventRoundTrip(t *testing.T) {
	b, err := encodeSession(&storedSession{ID: "s1", Events: storedEvents{{ID: "e1", LLMResponse: model.LLMResponse{Partial: true}}}})
	if err != nil {
		t.Fatalf("encodeSession() error = %v", err)
	}

	stored, err := decodeSession(string(b))
	if err != nil {
		t.Fatalf("decodeSession(%s) error = %v", b, err)
	}
	if len(stored.Events) != 1 {
		t.Fatalf("decodeSession(%s) returned %d events, want 1", b, len(stored.Events))
	}
	if got := stored.Events[0]; !got.Partial || !got.PartialSet {
		t.Errorf("decoded event = {Partial: %t, PartialSet: %t}, want {Partial: true, PartialSet: true}", got.Partial, got.PartialSet)
	}
}
