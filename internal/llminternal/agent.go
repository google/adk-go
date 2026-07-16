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

type State struct {
	Model model.LLM

	Mode Mode

	Tools    []tool.Tool
	Toolsets []tool.Toolset

	IncludeContents string

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

	// modeInit serializes ModeUnset -> default resolution so parallel
	// dispatches of a shared agent do not race on Mode.
	modeInit sync.Once
}

type InstructionProvider func(ctx agent.ReadonlyContext) (string, error)

func (s *State) internal() *State { return s }

// ResolveMode returns the agent's mode, defaulting ModeUnset exactly once
// to defaultIfUnset. Safe for concurrent callers sharing this State.
func (s *State) ResolveMode(defaultIfUnset Mode) Mode {
	s.modeInit.Do(func() {
		if s.Mode == ModeUnset {
			s.Mode = defaultIfUnset
		}
	})
	return s.Mode
}

func Reveal(a Agent) *State { return a.internal() }
