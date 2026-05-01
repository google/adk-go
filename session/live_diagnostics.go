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

package session

import "time"

// LiveDiagnostics captures computed timing and protocol state for a live
// streaming event. Only populated for events produced by RunLive; nil for
// standard Run() events.
//
// EPHEMERAL: This struct is NOT persisted to storage. It is attached to
// yielded events for the caller's use only. The runner strips it before
// AppendEvent and reattaches it before yielding.
type LiveDiagnostics struct {
	// --- Computed timing ---

	// TimeSinceLastEvent is the duration between this event's Timestamp
	// and the previous event's Timestamp.
	TimeSinceLastEvent time.Duration

	// TimeSinceLastSend is the duration between this event's ReceivedAt
	// and the most recent conn.Send (across all send paths: queued sends,
	// history replay, and tool responses).
	TimeSinceLastSend time.Duration

	// ReceiveLatency is the wall-clock duration of the blocking
	// conn.Receive() call that produced this event's LLMResponse.
	// Low = rapid streaming (audio chunks, partial text).
	// High = server thinking or waiting for input.
	// Zero for events not produced by conn.Receive (e.g. function-call events).
	ReceiveLatency time.Duration

	// ToolExecutionTime is the wall-clock duration from receiving the tool
	// call batch to sending the tool response. Zero for non-tool-response events.
	ToolExecutionTime time.Duration

	// QueueDepth is the number of LiveRequests buffered in the send queue
	// at the time this event was created.
	QueueDepth int

	// --- Protocol state (from Gemini Live API) ---

	// TurnCompleteReason from genai.TurnCompleteReason.
	// Empty when TurnComplete is false or reason is unspecified.
	TurnCompleteReason string

	// GoAwayTimeLeft is the server's remaining time before disconnect.
	// Zero when no GoAway message was received.
	GoAwayTimeLeft time.Duration

	// VADSignalType from genai.VoiceActivityDetectionSignal.
	// Empty when not present.
	VADSignalType string

	// SessionResumptionHandle is the opaque handle from
	// genai.LiveServerSessionResumptionUpdate.NewHandle for reconnection.
	SessionResumptionHandle string

	// SessionResumable from genai.LiveServerSessionResumptionUpdate.Resumable.
	SessionResumable bool

	// WaitingForInput is true when the server signals it needs user input.
	WaitingForInput bool

	// ModelSpeaking is true when the model is actively producing audio output.
	ModelSpeaking bool
}
