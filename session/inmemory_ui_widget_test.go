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

package session_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/session"
)

// The stored event keeps the widgets it was appended with: writes to the
// caller's slice and to a widget's payload map after the append do not reach
// it.
func TestInMemoryService_AppendEvent_CopiesRenderUIWidgets(t *testing.T) {
	ctx := t.Context()
	service := session.InMemoryService()
	createResp, err := service.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	event := &session.Event{ID: "e1", Author: "agent", Actions: session.EventActions{
		RenderUIWidgets: []session.UIWidget{{ID: "w1", Provider: "mcp", Payload: map[string]any{"k": "v1"}}},
	}}
	if err := service.AppendEvent(ctx, createResp.Session, event); err != nil {
		t.Fatalf("AppendEvent: %v", err)
	}

	event.Actions.RenderUIWidgets[0].ID = "mutated"
	event.Actions.RenderUIWidgets[0].Payload["k"] = "mutated"

	got, err := service.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: createResp.Session.ID()})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	want := []session.UIWidget{{ID: "w1", Provider: "mcp", Payload: map[string]any{"k": "v1"}}}
	if diff := cmp.Diff(want, got.Session.Events().At(0).Actions.RenderUIWidgets); diff != "" {
		t.Errorf("stored RenderUIWidgets mismatch (-want +got):\n%s", diff)
	}
}
