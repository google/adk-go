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

	// TaskCompletedInjection guards the one-time injection of the
	// task_completed tool and its instruction suffix that
	// sequentialagent.RunLive performs on each LLM sub-agent. Reveal
	// returns the same *State for the lifetime of an agent, so a tree
	// built once and driven by several concurrent live sessions would
	// otherwise race the read-then-append on Tools/Instruction. Keeping
	// the guard on State ties its lifetime to the agent's, so it is
	// collected with the agent and never accumulates in a process-global
	// map. It is only ever used via (*State), never a copy.
	TaskCompletedInjection sync.Once
}

type InstructionProvider func(ctx agent.ReadonlyContext) (string, error)

func (s *State) internal() *State { return s }

func Reveal(a Agent) *State { return a.internal() }
