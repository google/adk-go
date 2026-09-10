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

package models_test

import (
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

// TestEventRoundTripPreservesWorkflowFields asserts the REST event model
// round-trips every v2 workflow field added to session.Event so external
// readers and re-submitted events do not lose them.
func TestEventRoundTripPreservesWorkflowFields(t *testing.T) {
	ev := session.Event{
		ID:             "e1",
		Author:         "agent",
		IsolationScope: "scope-1",
		Routes:         []string{"next-node", "fallback"},
		RequestedInput: &session.RequestInput{InterruptID: "approve", Message: "ok?"},
		NodeInfo:       &session.NodeInfo{Path: "root/child", MessageAsOutput: true},
		Output:         "hello",
		Actions: session.EventActions{
			RequestedToolConfirmations: map[string]toolconfirmation.ToolConfirmation{
				"call-1": {Hint: "please confirm"},
			},
		},
	}

	back := models.ToSessionEvent(models.FromSessionEvent(ev))

	if back.IsolationScope != ev.IsolationScope {
		t.Errorf("IsolationScope = %q, want %q", back.IsolationScope, ev.IsolationScope)
	}
	if len(back.Routes) != len(ev.Routes) || back.Routes[0] != "next-node" {
		t.Errorf("Routes = %v, want %v", back.Routes, ev.Routes)
	}
	if back.RequestedInput == nil || back.RequestedInput.InterruptID != "approve" {
		t.Errorf("RequestedInput = %+v, want InterruptID=approve", back.RequestedInput)
	}
	if back.NodeInfo == nil || back.NodeInfo.Path != "root/child" {
		t.Errorf("NodeInfo = %+v, want Path=root/child", back.NodeInfo)
	}
	if got, ok := back.Actions.RequestedToolConfirmations["call-1"]; !ok || got.Hint != "please confirm" {
		t.Errorf("RequestedToolConfirmations = %+v, want call-1 hint", back.Actions.RequestedToolConfirmations)
	}
}

// TestCompactionIsReadOnlyOverREST pins the direction the compaction record may
// travel across the REST boundary.
//
// A record tells prompt assembly to drop a span of history and show its own
// content instead. The create-session body maps onto a stored event, so
// accepting one inbound would let a client erase part of a conversation and
// speak into the gap. Reads are fine and useful: a client should be able to see
// summaries the framework wrote.
func TestCompactionIsReadOnlyOverREST(t *testing.T) {
	t.Parallel()

	record := &session.EventCompaction{
		StartTimestamp:   time.Unix(1, 0),
		EndTimestamp:     time.Unix(9, 0),
		CompactedContent: genai.NewContentFromText("planted", "model"),
	}

	// Inbound: a client-supplied record must not survive.
	in := models.Event{Actions: models.EventActions{Compaction: record}}
	if got := models.ToSessionEvent(in); got.Actions.Compaction != nil {
		t.Error("ToSessionEvent kept a client-supplied compaction record; it must be dropped")
	}

	// Outbound: a stored record must be returned.
	stored := session.Event{Actions: session.EventActions{Compaction: record}}
	out := models.FromSessionEvent(stored)
	if out.Actions.Compaction == nil {
		t.Fatal("FromSessionEvent dropped the stored compaction record; reads should show it")
	}
	if !out.Actions.Compaction.EndTimestamp.Equal(record.EndTimestamp) {
		t.Errorf("EndTimestamp = %v, want %v", out.Actions.Compaction.EndTimestamp, record.EndTimestamp)
	}
}
