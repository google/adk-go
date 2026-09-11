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

package agent

import "errors"

// ErrLLMCallsLimitExceeded is returned when an invocation makes more model
// calls than RunConfig.MaxLLMCalls allows. Detect it with errors.Is.
var ErrLLMCallsLimitExceeded = errors.New("max number of llm calls exceeded")

// StreamingMode defines the streaming mode for agent execution.
type StreamingMode string

const (
	// StreamingModeNone indicates no streaming.
	StreamingModeNone StreamingMode = "none"
	// StreamingModeSSE enables server-sent events streaming, one-way, where
	// LLM response parts are streamed immediately as they are generated.
	StreamingModeSSE StreamingMode = "sse"
)

// RunConfig controls runtime behavior of an agent.
type RunConfig struct {
	// StreamingMode defines the streaming mode for an agent.
	StreamingMode StreamingMode
	// If true, ADK runner will save each part of the user input that is a blob
	// (e.g., images, files) as an artifact.
	SaveInputBlobsAsArtifacts bool
	// MaxLLMCalls bounds the total number of model calls one invocation may
	// make, across every agent it runs. Exceeding it ends the run with an error
	// wrapping ErrLLMCallsLimitExceeded.
	//
	// Zero, the value a caller gets from agent.RunConfig{}, means the default:
	// 500, or the value of the ADK_MAX_LLM_CALLS environment variable if it is
	// set to a valid integer. A negative value means no limit.
	//
	// The limit exists because whether a run terminates otherwise depends
	// entirely on model behaviour: a model that keeps requesting tool calls
	// appends an event per turn and re-sends a growing history, so token cost
	// grows quadratically with no exit. Mirrors adk-python's
	// RunConfig.max_llm_calls.
	MaxLLMCalls int
}
