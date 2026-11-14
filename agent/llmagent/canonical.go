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

package llmagent

import (
	"context"
	"errors"
	"fmt"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

// The built-in default names resolve through the model registry to ADK's
// Gemini connector, whose patterns package model/gemini registers when a
// program imports it. This package does not import model/gemini itself
// (model/gemini's own tests reach this package, which would make an import
// cycle), so a program that relies on a built-in default and builds no
// Gemini model imports it for its side effect:
//
//	import _ "google.golang.org/adk/v2/model/gemini"
//
// Without it, CanonicalModel and a run of a model-less agent report that no
// registered model matches the default name.
const (
	// DefaultModel is the name of the process-wide default model until a
	// setter changes it.
	DefaultModel = llminternal.DefaultModelName
	// DefaultLiveModel is the name of the process-wide default live model
	// until a setter changes it.
	DefaultLiveModel = llminternal.DefaultLiveModelName
)

// errNotLLMAgent is returned by the canonical accessors for an agent that
// llmagent.New did not build.
var errNotLLMAgent = errors.New("not an LLM agent")

// CanonicalModel returns the model the LLM agent a uses for a normal run
// when read outside a run: its own Config.Model, else the model of its
// nearest LLM-agent ancestor (non-LLM ancestors are skipped, an ancestor
// without a model inherits in turn), else the process-wide default model.
//
// Caller provides: an agent built by llmagent.New.
// Guarantees: the returned model is the object a run of a would call; the
// default is resolved through the model registry on first use and reused
// until a SetDefaultModel* call changes it.
// On failure: an error wrapping model.NewLLM's when the default name does not
// resolve; an error when a is nil or not an LLM agent.
// Zero value: N/A.
// Concurrency: safe for concurrent use.
//
// The built-in default name resolves only in a program that imports package
// model/gemini, which registers the Gemini patterns (see [DefaultModel]):
//
//	import _ "google.golang.org/adk/v2/model/gemini"
func CanonicalModel(a agent.Agent) (model.LLM, error) {
	return llminternal.ResolveModel(context.Background(), a)
}

// CanonicalLiveModel is CanonicalModel for a live run: the walk is the same,
// and the fallback is the process-wide default live model. As for
// CanonicalModel, the built-in default name resolves only in a program that
// imports package model/gemini:
//
//	import _ "google.golang.org/adk/v2/model/gemini"
func CanonicalLiveModel(a agent.Agent) (model.LLM, error) {
	return llminternal.ResolveLiveModel(context.Background(), a)
}

// CanonicalInstruction returns the instruction of the LLM agent a for ctx and
// whether the placeholders are left as they are: Config.InstructionProvider's
// result with true when a provider is set (it wins over Config.Instruction),
// else Config.Instruction with false.
//
// Caller provides: an LLM agent and a read-only context for the provider.
// Guarantees: the text exactly as configured or provided.
// On failure: the provider's error (wrapped) with "" and false; an error when
// a is nil or not an LLM agent.
// Zero value: N/A.
// Concurrency: safe for concurrent use.
func CanonicalInstruction(ctx agent.ReadonlyContext, a agent.Agent) (string, bool, error) {
	llmAgent, ok := a.(llminternal.Agent)
	if !ok {
		return "", false, fmt.Errorf("canonical instruction: %w", errNotLLMAgent)
	}
	state := llminternal.Reveal(llmAgent)
	if state.InstructionProvider == nil {
		return state.Instruction, false, nil
	}
	text, err := state.InstructionProvider(ctx)
	if err != nil {
		return "", false, fmt.Errorf("canonical instruction of agent %q: %w", a.Name(), err)
	}
	return text, true, nil
}

// CanonicalTools returns the tools of the LLM agent a for ctx: Config.Tools,
// then the tools each of Config.Toolsets lists, in that order; tool.AsToolset
// is the way to put a plain tool between toolsets. Toolsets are listed
// concurrently, each with ctx, and land in their declared position. A toolset
// whose listing fails contributes no tools and the call goes on, and no log
// line is written; but when ctx is not nil and is canceled (or past its
// deadline) after a listing has failed, the call returns an error that wraps
// both the toolset's failure and ctx.Err().
//
// The tools llmagent.New adds come after Config.Tools and before the
// toolsets' tools: for a task agent (Mode is ModeTask) the finish_task tool,
// then one agent tool for each sub-agent whose Mode is ModeSingleTurn or
// ModeTask, in sub-agent order. A ModeTask agent with Config.Tools [own] and
// one toolset listing [ts_a] returns [own finish_task ts_a]. adk-python
// offers these added tools after the toolsets' tools.
//
// Caller provides: an LLM agent (built by llmagent.New) and a read-only
// context; ctx may be nil, and toolsets then receive nil.
// Guarantees: a new slice that the caller may modify; the list a step of a
// run of a offers its model, as far as the toolsets answer the same way.
// On failure: the error above; an error when a is nil or not an LLM agent. A
// panic in a toolset's Tools reaches the caller.
// Zero value: N/A.
// Concurrency: safe for concurrent use; toolsets are called concurrently
// with the same ctx, so a ctx handed to CanonicalTools must be safe for that.
func CanonicalTools(ctx agent.ReadonlyContext, a agent.Agent) ([]tool.Tool, error) {
	llmAgent, ok := a.(llminternal.Agent)
	if !ok {
		return nil, fmt.Errorf("canonical tools: %w", errNotLLMAgent)
	}
	tools, err := llminternal.ResolveTools(ctx, llmAgent)
	if err != nil {
		return nil, fmt.Errorf("canonical tools of agent %q: %w", a.Name(), err)
	}
	return tools, nil
}

// SetDefaultModel sets the process-wide default model to m. It applies at
// once to every model-less agent that falls through to the default, built
// or not.
//
// Caller provides: a non-nil model.
// Guarantees: later resolutions return m itself.
// On failure: an error and no change when m is nil.
// Zero value: N/A.
// Concurrency: safe for concurrent use.
func SetDefaultModel(m model.LLM) error { return llminternal.SetDefaultModel(m) }

// SetDefaultModelByName sets the process-wide default model by name. The name
// is resolved through model.NewLLM on first use, and the resulting model is
// reused until a setter changes the default.
//
// Caller provides: a non-empty name.
// On failure: an error and no change when name is empty; an unknown name is
// reported by CanonicalModel, not here.
// Zero value: N/A.
// Concurrency: safe for concurrent use.
func SetDefaultModelByName(name string) error { return llminternal.SetDefaultModelByName(name) }

// SetDefaultLiveModel is SetDefaultModel for live runs.
func SetDefaultLiveModel(m model.LLM) error { return llminternal.SetDefaultLiveModel(m) }

// SetDefaultLiveModelByName is SetDefaultModelByName for live runs.
func SetDefaultLiveModelByName(name string) error {
	return llminternal.SetDefaultLiveModelByName(name)
}
