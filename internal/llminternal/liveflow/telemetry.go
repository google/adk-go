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

package liveflow

import (
	"sync"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

type liveTimingState struct {
	mu            sync.Mutex
	lastEventTime time.Time
	lastSendTime  time.Time
}

func (s *liveTimingState) recordSend(t time.Time) {
	s.mu.Lock()
	s.lastSendTime = t
	s.mu.Unlock()
}

// buildBaseDiagnostics creates a LiveDiagnostics with timing fields that
// apply to all event types (function-call, function-response, and content).
func buildBaseDiagnostics(
	queue *agent.LiveRequestQueue,
	ts *liveTimingState,
	receivedAt time.Time,
	eventTime time.Time,
) *session.LiveDiagnostics {
	diag := &session.LiveDiagnostics{
		ModelSpeaking: queue.ModelSpeaking(),
		QueueDepth:    queue.Len(),
	}

	ts.mu.Lock()
	if !ts.lastEventTime.IsZero() {
		diag.TimeSinceLastEvent = eventTime.Sub(ts.lastEventTime)
	}
	ts.lastEventTime = eventTime
	if !ts.lastSendTime.IsZero() && !receivedAt.IsZero() {
		diag.TimeSinceLastSend = receivedAt.Sub(ts.lastSendTime)
	}
	ts.mu.Unlock()

	return diag
}

// populateProtocolState extracts protocol state from CustomMetadata into LiveDiagnostics.
func populateProtocolState(resp *model.LLMResponse, diag *session.LiveDiagnostics) {
	if reason, ok := resp.CustomMetadata["turn_complete_reason"].(string); ok {
		diag.TurnCompleteReason = reason
	}
	if ms, ok := resp.CustomMetadata["go_away_time_left_ms"].(float64); ok {
		diag.GoAwayTimeLeft = time.Duration(ms) * time.Millisecond
	}
	if vad, ok := resp.CustomMetadata["vad_signal_type"].(string); ok {
		diag.VADSignalType = vad
	}
	if handle, ok := resp.CustomMetadata["session_resumption_handle"].(string); ok {
		diag.SessionResumptionHandle = handle
	}
	if resumable, ok := resp.CustomMetadata["session_resumption_resumable"].(bool); ok {
		diag.SessionResumable = resumable
	}
	if wi, ok := resp.CustomMetadata["waiting_for_input"].(bool); ok {
		diag.WaitingForInput = wi
	}
}
