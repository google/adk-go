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

package agent

// holds Agent internal state
type Agent interface {
	internal() *State
}

type State struct {
	AgentType Type
	Config    any
}

type Type string

const (
	TypeLLMAgent        Type = "LLMAgent"
	TypeLoopAgent       Type = "LoopAgent"
	TypeSequentialAgent Type = "SequentialAgent"
	TypeParallelAgent   Type = "ParallelAgent"
	TypeCustomAgent     Type = "CustomAgent"
)

func (s *State) internal() *State { return s }

func Reveal(a Agent) *State { return a.internal() }
