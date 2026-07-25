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

	// resolveOnce guards the one-time, race-free resolution of Mode and the
	// derived IncludeContents default. See ResolveMode.
	resolveOnce sync.Once
}

type InstructionProvider func(ctx agent.ReadonlyContext) (string, error)

func (s *State) internal() *State { return s }

func Reveal(a Agent) *State { return a.internal() }

// ResolveMode resolves the agent's run Mode exactly once: an unset Mode is
// defaulted to def, and single_turn agents get IncludeContents "none". The
// default is context-dependent (chat for a root agent, single_turn for a
// sub-agent node), so it must be resolved at first run rather than at
// construction.
//
// It is safe for concurrent callers: dispatching the same sub-agent on parallel
// branches previously raced on these writes (google/adk-go#1137). The sync.Once
// both dedupes the write and establishes a happens-before edge, so after any
// call returns, Mode and IncludeContents are stable and may be read without
// further synchronization.
func (s *State) ResolveMode(def Mode) {
	s.resolveOnce.Do(func() {
		if s.Mode == ModeUnset {
			s.Mode = def
		}
		if s.Mode == ModeSingleTurn {
			s.IncludeContents = "none"
		}
	})
}
