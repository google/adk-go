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

package replayplugin

import (
	"sync"

	"google.golang.org/adk/internal/configurable/conformance/replayplugin/recording"
)

// invocationReplayState tracks per-invocation replay state to isolate concurrent runs.
type invocationReplayState struct {
	testCasePath     string
	userMessageIndex int
	recordings       *recording.Recordings

	// Per-agent replay indices for parallel execution
	// key: agent_name -> current replay index for that agent
	agentReplayIndices map[string]int

	curIndex int
	mu       sync.Mutex
	cond     *sync.Cond
}

// newInvocationReplayState behaves as the constructor.
func newInvocationReplayState(testCasePath string, userMessageIndex int, recs *recording.Recordings) *invocationReplayState {
	state := &invocationReplayState{
		testCasePath:       testCasePath,
		userMessageIndex:   userMessageIndex,
		recordings:         recs,
		agentReplayIndices: make(map[string]int),
		curIndex:           0,
		mu:                 sync.Mutex{},
	}
	state.cond = sync.NewCond(&state.mu)
	return state
}

// GetTestCasePath returns the test case path.
func (s *invocationReplayState) GetTestCasePath() string {
	return s.testCasePath
}

// GetUserMessageIndex returns the user message index.
func (s *invocationReplayState) GetUserMessageIndex() int {
	return s.userMessageIndex
}

// GetRecordings returns the recordings object.
func (s *invocationReplayState) GetRecordings() *recording.Recordings {
	return s.recordings
}

// GetAgentReplayIndex returns the index for the agent.
// In Go, looking up a missing key returns the zero value (0),
// so getOrDefault is intrinsic to the language for integers.
func (s *invocationReplayState) GetAgentReplayIndex(agentName string) (int, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	val, ok := s.agentReplayIndices[agentName]
	return val, ok
}
