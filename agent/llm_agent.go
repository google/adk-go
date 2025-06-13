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

import (
	"context"

	"github.com/google/adk-go"
	"google.golang.org/genai"
)

// LLMAgent is an LLM-based Agent.
type LLMAgent struct {
	AgentName        string
	AgentDescription string

	Model adk.Model

	Instruction           string
	GlobalInstruction     string
	Tools                 []adk.Tool
	GenerateContentConfig *genai.GenerateContentConfig

	// LLM-based agent transfer configs.
	DisallowTransferToParent bool
	DisallowTransferToPeers  bool

	// BeforeModelCallback
	// AfterModelCallback
	// BeforeToolCallback
	// AfterToolCallback
}

func (a *LLMAgent) Name() string        { return a.AgentName }
func (a *LLMAgent) Description() string { return a.AgentDescription }
func (a *LLMAgent) Run(ctx context.Context, parentCtx *adk.InvocationContext) (adk.EventStream, error) {
	// TODO: Select model (LlmAgent.canonical_model)
	// TODO: Singleflow, Autoflow Run.
	panic("unimplemented")
}

var _ adk.Agent = (*LLMAgent)(nil)

// TODO: Do we want to abstract "Flow" too?
