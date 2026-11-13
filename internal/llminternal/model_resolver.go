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

package llminternal

import (
	"context"
	"fmt"

	"google.golang.org/adk/v2/agent"
	agentinternal "google.golang.org/adk/v2/internal/agent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	"google.golang.org/adk/v2/model"
)

// ResolveModel returns the model a normal run of the LLM agent a calls, as
// adk-python's LlmAgent.canonical_model resolves it: a's own model, else the
// model of its nearest LLM-agent ancestor that has one (other agents in
// between are skipped), else the process-wide default model.
//
// Inside a run, ctx carries the run's parent map and the ancestors are read
// from the tree being run; from the run's root upward, and outside a run
// (context.Background()), they are read from the parent links agent.New
// records. It returns an error wrapping ErrModelNotConfigured when a is not
// an LLM agent, and the registry's error when the default name does not
// resolve.
func ResolveModel(ctx context.Context, a agent.Agent) (model.LLM, error) {
	return resolveModel(ctx, a, defaultNormalModel)
}

// ResolveLiveModel is ResolveModel for a live run: the walk is the same and
// the fallback is the process-wide default live model.
func ResolveLiveModel(ctx context.Context, a agent.Agent) (model.LLM, error) {
	return resolveModel(ctx, a, defaultLiveModel)
}

func resolveModel(ctx context.Context, a agent.Agent, fallback *defaultModel) (model.LLM, error) {
	llmAgent := asLLMAgent(a)
	if llmAgent == nil {
		return nil, fmt.Errorf("resolve the model of %s: not an LLM agent: %w", describeAgent(a), ErrModelNotConfigured)
	}
	if m := llmAgent.internal().Model; m != nil {
		return m, nil
	}
	if m := ancestorModel(ctx, a); m != nil {
		return m, nil
	}
	m, err := fallback.resolve(ctx)
	if err != nil {
		return nil, fmt.Errorf("resolve the model of %s: %w", describeAgent(a), err)
	}
	return m, nil
}

// setStepModel sets the model of the step that is starting: f.Model when
// set, else the model resolve returns for ctx's agent.
func (f *Flow) setStepModel(ctx agent.InvocationContext, resolve func(context.Context, agent.Agent) (model.LLM, error)) error {
	if f.Model != nil {
		f.stepModel = f.Model
		return nil
	}
	m, err := resolve(ctx, ctx.Agent())
	if err != nil {
		return err
	}
	f.stepModel = m
	return nil
}

// currentModel returns the model of the current step, or Model for a Flow
// method called outside a step, as unit tests call callLLM.
func (f *Flow) currentModel() model.LLM {
	if f.stepModel != nil {
		return f.stepModel
	}
	return f.Model
}

// processorModel returns the model a request processor reads: the step's
// model, else f.Model, else the resolved model of ctx's agent. The last two
// serve processors called outside a step, as unit tests call them.
func processorModel(ctx agent.InvocationContext, f *Flow) model.LLM {
	if f != nil {
		if m := f.currentModel(); m != nil {
			return m
		}
	}
	m, err := ResolveModel(ctx, ctx.Agent())
	if err != nil {
		// Outside a step nothing calls the model, so there is no step to
		// fail: a model that does not resolve reads as no model, as a nil
		// Flow.Model did.
		return nil
	}
	return m
}

// ancestorModel returns the model of the nearest LLM-agent ancestor of a
// that has one, or nil.
func ancestorModel(ctx context.Context, a agent.Agent) model.LLM {
	cur := a
	if parents := parentmap.FromContext(ctx); parents != nil {
		for {
			parent := parents[cur.Name()]
			if parent == nil {
				break // cur is the run's root
			}
			if m := ownModel(parent); m != nil {
				return m
			}
			cur = parent
		}
	}
	return staticAncestorModel(innerAgent(cur))
}

// staticAncestorModel follows the parent links agent.New records, starting
// at the parent of inner. It reads them through agentinternal.ParentOf, so
// it is safe while another goroutine constructs a parent of an agent on the
// walk.
func staticAncestorModel(inner agentinternal.Agent) model.LLM {
	for inner != nil {
		parent, owner := agentinternal.ParentOf(agentinternal.Reveal(inner))
		if parent == nil {
			return nil
		}
		if owner, ok := owner.(agent.Agent); ok {
			if m := ownModel(owner); m != nil {
				return m
			}
		}
		inner = parent
	}
	return nil
}

// ownModel returns the model configured on a when a is an LLM agent, or a
// value that embeds one, such as a wrapper that overrides FindAgent.
func ownModel(a agent.Agent) model.LLM {
	if llmAgent := asLLMAgent(a); llmAgent != nil {
		return llmAgent.internal().Model
	}
	if owner, ok := agentinternal.OwnerOf(agentinternal.InnerState(a)).(agent.Agent); ok {
		if llmAgent := asLLMAgent(owner); llmAgent != nil {
			return llmAgent.internal().Model
		}
	}
	return nil
}

// innerAgent returns the inner agent that holds a's parent link: the base
// agent of an LLM agent, else the agent agent.New built that a is or embeds.
func innerAgent(a agent.Agent) agentinternal.Agent {
	if llmAgent := asLLMAgent(a); llmAgent != nil {
		return llmAgent.internal().Base
	}
	return agentinternal.InnerState(a)
}

func describeAgent(a agent.Agent) string {
	if a == nil {
		return "a nil agent"
	}
	return fmt.Sprintf("agent %q", a.Name())
}
