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

package recordplugin

import (
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/internal/configurable/conformance/replayplugin/recording"
	"google.golang.org/adk/model"
)

type invocationRecordState struct {
	mu               sync.Mutex
	caseDir          string
	userMessageIndex int
	streamingMode    string

	// Completed recordings for the current invocation turn
	recordings []recording.Recording

	// Active/pending LLM request mappings
	pendingLLMRequest   *model.LLMRequest
	pendingLLMResponses []*model.LLMResponse

	// Active/pending Tool call mappings keyed by function_call_id
	pendingToolCalls map[string]*genai.FunctionCall
}

func newInvocationRecordState(caseDir string, msgIndex int, streamingMode string) *invocationRecordState {
	return &invocationRecordState{
		caseDir:          caseDir,
		userMessageIndex: msgIndex,
		streamingMode:    streamingMode,
		pendingToolCalls: make(map[string]*genai.FunctionCall),
	}
}

func (s *invocationRecordState) StartLLMRecording(req *model.LLMRequest) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.pendingLLMRequest = req
	s.pendingLLMResponses = nil
}

func (s *invocationRecordState) AppendLLMResponse(resp *model.LLMResponse) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.pendingLLMResponses = append(s.pendingLLMResponses, resp)
}

func (s *invocationRecordState) CompleteLLMRecording(agentName string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.pendingLLMRequest == nil {
		return
	}

	s.recordings = append(s.recordings, recording.Recording{
		UserMessageIndex: s.userMessageIndex,
		AgentName:        agentName,
		LLMRecording: &recording.LLMRecording{
			LLMRequest:   s.pendingLLMRequest,
			LLMResponses: s.pendingLLMResponses,
		},
	})

	s.pendingLLMRequest = nil
	s.pendingLLMResponses = nil
}

func (s *invocationRecordState) StartToolRecording(id string, fc *genai.FunctionCall) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.pendingToolCalls[id] = fc
}

func (s *invocationRecordState) CompleteToolRecording(id, agentName string, resp *genai.FunctionResponse) {
	s.mu.Lock()
	defer s.mu.Unlock()

	fc, ok := s.pendingToolCalls[id]
	if !ok {
		return
	}

	s.recordings = append(s.recordings, recording.Recording{
		UserMessageIndex: s.userMessageIndex,
		AgentName:        agentName,
		ToolRecording: &recording.ToolRecording{
			ToolCall:     fc,
			ToolResponse: resp,
		},
	})

	delete(s.pendingToolCalls, id)
}
