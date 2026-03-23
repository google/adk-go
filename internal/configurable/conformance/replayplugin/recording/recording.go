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

package recording

import (
	"google.golang.org/genai"

	"google.golang.org/adk/model"
)

// Recordings represents all recordings in chronological order.
type Recordings struct {
	// Chronological list of all recordings.
	Recordings []Recording `yaml:"recordings"`
}

// Recording represents a single interaction recording, ordered by request timestamp.
type Recording struct {
	// Index of the user message this recording belongs to (0-based).
	UserMessageIndex int `yaml:"usermessageindex"`

	// Name of the agent.
	AgentName string `yaml:"agentname"`

	// oneof fields - start

	// LLM request-response pair.
	LLMRecording *LLMRecording `yaml:"llmrecording,omitempty"`

	// Tool call-response pair.
	ToolRecording *ToolRecording `yaml:"toolrecording,omitempty"`

	// oneof fields - end

	// Index of the recording in the recordings list (0-based).
	Index int `yaml:"-"`
}

// LLMRecording represents a paired LLM request and response.
type LLMRecording struct {
	// Required. The LLM request.
	LLMRequest *model.LLMRequest `yaml:"llmrequest,omitempty"`

	// Required. The LLM response.
	LLMResponses []*model.LLMResponse `yaml:"llmresponses,omitempty"`
}

// ToolRecording represents a paired tool call and response.
type ToolRecording struct {
	// Required. The tool call.
	ToolCall *genai.FunctionCall `yaml:"toolcall,omitempty"`

	// Required. The tool response.
	ToolResponse *genai.FunctionResponse `yaml:"toolresponse,omitempty"`
}
