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
	"context"

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

type defaultModeKey struct{}

// defaultMode is the delegation mode to assume for one named agent whose
// State.Mode is ModeUnset.
type defaultMode struct {
	agent string
	mode  Mode
}

// WithDefaultMode returns ctx carrying mode as the delegation mode to assume
// for the agent named agentName when its own Mode is unset.
//
// Such a default depends on the agent's role in one particular call: the
// runner's root agent must be a chat agent, an agent wrapped as a workflow node
// defaults to single_turn. State, meanwhile, is shared by every concurrent
// invocation of that agent, so writing the default into it is a data race
// (issue #1137) — the role travels with the call instead. The declaration
// covers that one agent, not the agents it dispatches: whoever dispatches them
// declares their role.
func WithDefaultMode(ctx context.Context, agentName string, mode Mode) context.Context {
	return context.WithValue(ctx, defaultModeKey{}, defaultMode{agent: agentName, mode: mode})
}

// defaultModeFromContext returns the mode WithDefaultMode declared for
// agentName, or ModeUnset if ctx declares none for it.
func defaultModeFromContext(ctx context.Context, agentName string) Mode {
	d, _ := ctx.Value(defaultModeKey{}).(defaultMode)
	if d.agent != agentName {
		return ModeUnset
	}
	return d.mode
}

// ResolveMode returns the delegation mode to run the agent named agentName
// under, in precedence order: its configured Mode, the default its caller
// declared on ctx, then fallback. It never writes to s.
func (s *State) ResolveMode(ctx context.Context, agentName string, fallback Mode) Mode {
	if s.Mode != ModeUnset {
		return s.Mode
	}
	if mode := defaultModeFromContext(ctx, agentName); mode != ModeUnset {
		return mode
	}
	return fallback
}

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
}

type InstructionProvider func(ctx agent.ReadonlyContext) (string, error)

func (s *State) internal() *State { return s }

func Reveal(a Agent) *State { return a.internal() }
