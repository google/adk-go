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
	"fmt"
	"strings"
	"sync"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
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

	OutputKey      string
	OutputArtifact string

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

// OutputArtifactReference is the text that replaces an agent's final reply
// once the reply is saved as revision version of the artifact name.
func OutputArtifactReference(name string, version int64) string {
	return fmt.Sprintf("Saved artifact %q (version %d).", name, version)
}

// SavedOutputArtifact reports whether the agent already saved its reply on ev
// as s.OutputArtifact. A model callback can save the same filename on the same
// event, so a key in ArtifactDelta alone does not mark the agent's own save:
// the event's text must also be the reference to the recorded revision.
func (s *State) SavedOutputArtifact(ev *session.Event) bool {
	if s.OutputArtifact == "" || ev == nil || ev.Content == nil {
		return false
	}
	version, ok := ev.Actions.ArtifactDelta[s.OutputArtifact]
	if !ok {
		return false
	}
	var text strings.Builder
	for _, part := range ev.Content.Parts {
		if part != nil && !part.Thought {
			text.WriteString(part.Text)
		}
	}
	return text.String() == OutputArtifactReference(s.OutputArtifact, version)
}
