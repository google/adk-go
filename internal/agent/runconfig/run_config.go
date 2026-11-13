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

package runconfig

import (
	"context"
	"sync/atomic"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/adkcontext"
)

type StreamingMode string

const (
	StreamingModeNone StreamingMode = "none"
	StreamingModeSSE  StreamingMode = "sse"
	StreamingModeBidi StreamingMode = "bidi"
)

// RunConfig carries the per-invocation runtime settings that the flow needs but
// that are not part of any single agent's configuration. It is stored on the Go
// context by the runner and must be used by pointer: it holds the invocation's
// model-call counter.
type RunConfig struct {
	StreamingMode StreamingMode
	Live          *agent.LiveRunConfig

	// MaxLLMCalls is the resolved model-call budget for this invocation. See
	// ResolveMaxLLMCalls. Non-positive means no limit.
	MaxLLMCalls int

	// HTTPOptions is the invocation's agent.RunConfig.HTTPOptions. The flow
	// reads it from the invocation context; it is kept here for
	// tool/agenttool, whose tool context has no run config, to forward to the
	// nested run.
	HTTPOptions *genai.HTTPOptions

	// Labels is the invocation's agent.RunConfig.Labels, kept here for
	// tool/agenttool to forward to the nested run, as HTTPOptions is.
	Labels *map[string]string

	// CustomMetadata holds the invocation's pre-filtered custom metadata
	// (eventmetadata.WithoutInternal), or nil when none was supplied. It is
	// the runner's own copy: never write to it, and never hand it to an event
	// without copying.
	CustomMetadata map[string]any

	// Resumable reports whether the runner that installed this RunConfig is
	// resumable. A context without a RunConfig is not resumable.
	Resumable bool

	// AgentStates is the snapshot of agent states the runner rebuilt from the
	// session when it resumed the invocation on the classic path, keyed by
	// agent name. Nil otherwise. Read it, never write it.
	AgentStates map[string]map[string]any

	// NodePath reports whether the run entered through the runner's node
	// path (an LlmAgent or node root). Only that entry sets it. On a
	// resumable node-path run, an LlmAgent that transfers writes its
	// end-of-agent event before the transfer target runs and writes none
	// after; on the classic path it writes it after the target returns.
	NodePath bool

	// llmCalls counts model calls made during this invocation, across every
	// agent it runs, so one budget covers the whole invocation rather than one
	// budget per agent.
	llmCalls atomic.Int64
}

func ToContext(ctx context.Context, cfg *RunConfig) context.Context {
	// TODO(kdroste): move to invocationContext?
	return context.WithValue(ctx, adkcontext.RunConfigKey, cfg)
}

func FromContext(ctx context.Context) *RunConfig {
	m, ok := ctx.Value(adkcontext.RunConfigKey).(*RunConfig)
	if !ok {
		return nil
	}
	return m
}

// CustomMetadataSnapshot returns the invocation's pre-filtered custom metadata,
// the CustomMetadata field, or nil for a nil receiver. The map is the runner's
// own copy: callers must not write to it, and copy it before handing it on.
//
// It exists so that package agent, which cannot import this package, can read
// the snapshot from the value stored under [adkcontext.RunConfigKey].
func (c *RunConfig) CustomMetadataSnapshot() map[string]any {
	if c == nil {
		return nil
	}
	return c.CustomMetadata
}
