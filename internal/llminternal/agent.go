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

	// LiveInjection guards the single live-mode tool injection an agent may
	// receive: whichever workflow agent runs this one live first appends its
	// hand-off tool and instruction suffix to Tools/Instruction, once, and
	// every concurrent caller blocks until that has finished. Kept on State
	// so its lifetime is the agent's — collected with the agent, never
	// accumulated in a process-global map, and only ever used via (*State).
	// Today sequentialagent.RunLive is the only injector.
	LiveInjection LiveToolInjection
}

// LiveToolInjection is a one-shot, panic-safe latch for the single
// live-mode tool injection an agent may receive (see State.LiveInjection).
// The first caller to Do performs the injection; concurrent callers block
// and then observe it as a no-op, with a happens-before edge to the first
// caller's writes. Unlike sync.Once it does not latch when the injection
// panics, so a misconfigured sub-agent stays loud on every call — the same
// behaviour as re-deriving the decision each time — instead of going
// silent after the first failure. If a second workflow agent ever adds its
// own live-mode injection it shares this latch: first write wins, the rest
// are skipped.
type LiveToolInjection struct {
	mu   sync.Mutex
	done bool
}

// Do runs inject at most once for the lifetime of the receiver. It
// serialises concurrent callers, gives the ones that lose the race a
// happens-before edge to the winner's writes, and — because done is set
// only after inject returns normally — lets a panic propagate without
// latching, so a later call retries.
func (l *LiveToolInjection) Do(inject func()) {
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
