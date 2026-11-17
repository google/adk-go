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

package agentanalytics

import (
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestSessionEvents_AgentStateCheckpointFromTypedActions(t *testing.T) {
	isTrue, isFalse := true, false
	tests := []struct {
		name          string
		maxContentLen int
		actions       session.EventActions
		meta          map[string]any
		wantEmitted   bool
		wantJSON      string
		wantTruncated bool
	}{
		{
			name:        "agent state only",
			actions:     session.EventActions{AgentState: map[string]any{"step": 1}},
			wantEmitted: true,
			wantJSON:    `{"agent_state":{"step":1},"end_of_agent":false}`,
		},
		{
			name:        "empty agent state is a checkpoint",
			actions:     session.EventActions{AgentState: map[string]any{}},
			wantEmitted: true,
			wantJSON:    `{"agent_state":{},"end_of_agent":false}`,
		},
		{
			name:          "agent state strings are truncated",
			maxContentLen: 5,
			actions:       session.EventActions{AgentState: map[string]any{"cursor": "abcdefgh"}},
			wantEmitted:   true,
			wantJSON:      `{"agent_state":{"cursor":"abcde...[TRUNCATED]"},"end_of_agent":false}`,
			wantTruncated: true,
		},
		{
			name:        "end of agent only with nil state",
			actions:     session.EventActions{EndOfAgent: &isTrue},
			wantEmitted: true,
			wantJSON:    `{"agent_state":null,"end_of_agent":true}`,
		},
		{
			name:        "agent state and end of agent, typed fields win over metadata",
			actions:     session.EventActions{AgentState: map[string]any{"step": 2}, EndOfAgent: &isTrue},
			meta:        map[string]any{"adk_agent_state": map[string]any{"step": 9}, "end_of_agent": false},
			wantEmitted: true,
			wantJSON:    `{"agent_state":{"step":2},"end_of_agent":true}`,
		},
		{
			name:        "neither typed field nor metadata writes no row",
			wantEmitted: false,
		},
		{
			name:        "end of agent false with nil state writes no row even with metadata state",
			actions:     session.EventActions{EndOfAgent: &isFalse},
			meta:        map[string]any{"adk_agent_state": map[string]any{"step": 9}},
			wantEmitted: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
				if tt.maxContentLen != 0 {
					c.MaxContentLen = tt.maxContentLen
				}
				c.CustomMetadataAllowlist = []string{"adk_agent_state", "end_of_agent"}
			})
			var errs callbackErrors
			errs.add(p.OnEventCallback()(mCtx, &session.Event{
				ID:          "ev-typed-ckpt",
				Author:      "workflow-agent",
				Actions:     tt.actions,
				LLMResponse: model.LLMResponse{CustomMetadata: tt.meta},
			}))
			errs.check(t)
			p.AfterRunCallback()(mCtx)

			rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
			if !tt.wantEmitted {
				if got := rowsOfType(rows, EventAgentStateCheckpoint); len(got) != 0 {
					t.Fatalf("got %d AGENT_STATE_CHECKPOINT rows, want 0: %v", len(got), got)
				}
				return
			}
			row := onlyRowOfType(t, rows, EventAgentStateCheckpoint)
			checkJSON(t, "AGENT_STATE_CHECKPOINT content", jsonObjectColumn(t, row, "content"), tt.wantJSON)
			if got := row["is_truncated"]; got != tt.wantTruncated {
				t.Errorf("is_truncated = %v, want %v", got, tt.wantTruncated)
			}
		})
	}
}
