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

package llminternal

import (
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

// holds LLMAgent internal state
type Agent interface {
	internal() *State
}

type Mode string

const (
	ModeUnset      Mode = ""
	ModeChat       Mode = "chat"
	ModeTask       Mode = "task"
	ModeSingleTurn Mode = "single_turn"
)

// IncludeContents controls what parts of prior conversation history is received by llmagent.
type IncludeContents string

const (
	// IncludeContentsNone makes the llmagent operate solely on its current turn (latest user input + any following agent events).
	IncludeContentsNone IncludeContents = "none"
	// IncludeContentsDefault is enabled by default. The llmagent receives the relevant conversation history.
	IncludeContentsDefault IncludeContents = "default"
)

type State struct {
	Model model.LLM

	Mode Mode

	Tools    []tool.Tool
	Toolsets []tool.Toolset

	IncludeContents IncludeContents

	GenerateContentConfig *genai.GenerateContentConfig

	Instruction               string
	InstructionProvider       InstructionProvider
	GlobalInstruction         string
	GlobalInstructionProvider InstructionProvider

	DisallowTransferToParent bool
	DisallowTransferToPeers  bool

	InputSchema  *genai.Schema
	OutputSchema *genai.Schema

	OutputKey string

	// LiveModeInjection guards the single live-mode tool injection an agent
	// receives. Its lifetime follows the agent rather than a global cache.
	// Only callers of Do are synchronized; other State readers take no lock.
	LiveModeInjection LiveModeToolInjection
}

// LiveModeToolInjection allows one successful live-mode tool injection per
// agent. Multiple injectors share this latch, so the first successful one wins.
// A panicking injection is retried by a later caller.
// LiveModeToolInjection must not be copied after first use.
type LiveModeToolInjection struct {
	mu   sync.Mutex
	done bool
}

// Do runs inject at most once successfully. Concurrent callers wait for the
// injection to finish before returning. A panic propagates without marking the
// injection complete. inject must not block indefinitely or re-enter Do on the
// same receiver.
func (l *LiveModeToolInjection) Do(inject func()) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.done {
		return
	}
	inject()
	l.done = true
}

type InstructionProvider func(ctx agent.ReadonlyContext) (string, error)

func (s *State) internal() *State { return s }

func Reveal(a Agent) *State { return a.internal() }
