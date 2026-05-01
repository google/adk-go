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

// Package flowtool exposes a single tool, run_flow, that lets an LLM
// dynamically compose a catalog of pre-registered agents into sequential
// or parallel flows at runtime.
//
// The catalog is fixed at parent construction; the LLM only chooses the
// shape, the subset, and the per-node inputs.
//
// Example:
//
//	catalog := map[string]agent.Agent{
//	    "researcher":   researcher,
//	    "drafter":      drafter,
//	    "fact_checker": factChecker,
//	    "editor":       editor,
//	}
//	parent := llmagent.New(llmagent.Config{
//	    Name:  "orchestrator",
//	    Model: model,
//	    Tools: []tool.Tool{flowtool.New(catalog)},
//	    ...
//	})
package flowtool

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

// Default limits.
const (
	defaultMaxConcurrency   = 8
	defaultMaxNodes         = 32
	defaultMaxDepth         = 4
	defaultMaxParallelWidth = 8
	defaultTimeout          = 5 * time.Minute
	defaultMaxRecursion     = 3
)

// Option configures a flowTool.
type Option func(*flowTool)

// WithMaxConcurrency caps concurrent agent invocations inside a single
// parallel branch. Default 8.
func WithMaxConcurrency(n int) Option { return func(t *flowTool) { t.maxConcurrency = n } }

// WithMaxNodes caps the total node count in a single flow spec. Default 32.
func WithMaxNodes(n int) Option { return func(t *flowTool) { t.maxNodes = n } }

// WithMaxDepth caps spec nesting depth. Default 4.
func WithMaxDepth(n int) Option { return func(t *flowTool) { t.maxDepth = n } }

// WithMaxParallelWidth caps the fan-out width of any single parallel node.
// Default 8.
func WithMaxParallelWidth(n int) Option { return func(t *flowTool) { t.maxParallelWidth = n } }

// WithTimeout caps total wall-clock time for one run_flow invocation.
// Default 5m.
func WithTimeout(d time.Duration) Option { return func(t *flowTool) { t.timeout = d } }

// WithMaxRecursion caps how many run_flow calls may be nested in the same
// invocation chain. Default 3.
func WithMaxRecursion(n int) Option { return func(t *flowTool) { t.maxRecursion = n } }

// WithInheritState, when true, copies the parent invocation's session state
// (excluding _adk* keys) into each spawned sub-agent's session as a
// read-only seed. Writes from sub-agents do not propagate back. Default false.
func WithInheritState(v bool) Option { return func(t *flowTool) { t.inheritState = v } }

// WithName overrides the LLM-facing tool name (default "run_flow").
func WithName(name string) Option { return func(t *flowTool) { t.name = name } }

// New constructs a run_flow tool over the given catalog.
//
// The catalog maps names (as the LLM will refer to them) to concrete
// agent.Agent values. Pass an empty map to disable; the tool will refuse
// any spec.
func New(catalog map[string]agent.Agent, opts ...Option) tool.Tool {
	t := &flowTool{
		name:             "run_flow",
		catalog:          map[string]agent.Agent{},
		maxConcurrency:   defaultMaxConcurrency,
		maxNodes:         defaultMaxNodes,
		maxDepth:         defaultMaxDepth,
		maxParallelWidth: defaultMaxParallelWidth,
		timeout:          defaultTimeout,
		maxRecursion:     defaultMaxRecursion,
	}
	for k, v := range catalog {
		if v == nil {
			continue
		}
		t.catalog[k] = v
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

type flowTool struct {
	name    string
	catalog map[string]agent.Agent

	maxConcurrency   int
	maxNodes         int
	maxDepth         int
	maxParallelWidth int
	timeout          time.Duration
	maxRecursion     int
	inheritState     bool
}

// Name implements tool.Tool.
func (t *flowTool) Name() string { return t.name }

// Description implements tool.Tool.
func (t *flowTool) Description() string {
	names := make([]string, 0, len(t.catalog))
	for k := range t.catalog {
		names = append(names, k)
	}
	sort.Strings(names)
	var b strings.Builder
	b.WriteString("Compose registered agents into a sequential or parallel flow ")
	b.WriteString("and run it. The flow spec is recursive: each node is either ")
	b.WriteString(`{"type":"agent","agent":"<name>","input":"..."}, `)
	b.WriteString(`{"type":"seq","nodes":[...]}, or `)
	b.WriteString(`{"type":"parallel","nodes":[...]}. `)
	b.WriteString("In a seq, each node's output feeds the next. In a parallel, ")
	b.WriteString("siblings receive the same predecessor output. ")
	b.WriteString(`The optional "input" field on an agent node can include `)
	b.WriteString(`{{nodes.<path>.output}} templates referencing earlier outputs. `)
	b.WriteString("Available agents: ")
	if len(names) == 0 {
		b.WriteString("(none)")
	} else {
		b.WriteString(strings.Join(names, ", "))
	}
	b.WriteString(". Returns {outputs: {<path>: {output, error?}}, final_output: string}.")
	return b.String()
}

// IsLongRunning implements tool.Tool.
func (t *flowTool) IsLongRunning() bool { return false }

// Declaration returns the function declaration shown to the LLM.
func (t *flowTool) Declaration() *genai.FunctionDeclaration {
	specSchema := flowSpecSchema()
	return &genai.FunctionDeclaration{
		Name:        t.name,
		Description: t.Description(),
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"initial_input": {Type: "STRING", Description: "Input passed to the first node of the flow."},
				"spec":          specSchema,
			},
			Required: []string{"spec"},
		},
	}
}

// flowSpecSchema returns a permissive schema for the recursive spec. We
// keep it shallow (one level of nodes typed as OBJECT) so genai accepts
// the declaration without recursive $ref support; deeper nesting is still
// validated by UnmarshalJSON at runtime.
func flowSpecSchema() *genai.Schema {
	return &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"type":  {Type: "STRING", Description: "agent | seq | parallel"},
			"agent": {Type: "STRING", Description: "Catalog name (agent nodes only)."},
			"input": {Type: "STRING", Description: "Per-node input. Supports {{nodes.<path>.output}} templates."},
			"nodes": {
				Type:        "ARRAY",
				Description: "Child nodes (seq | parallel only). Each child has the same shape as this object.",
				Items:       &genai.Schema{Type: "OBJECT"},
			},
		},
		Required: []string{"type"},
	}
}

// Run executes the flow spec.
func (t *flowTool) Run(toolCtx tool.Context, args any) (map[string]any, error) {
	margs, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("flowtool: expected map[string]any args, got %T", args)
	}

	parentCtx, ok := toolCtx.(context.Context)
	if !ok {
		// tool.Context embeds context.Context indirectly via CallbackContext.
		// Fall back to background if the type assertion misses; in practice
		// the runner always supplies a context-bearing toolCtx.
		parentCtx = context.Background()
	}

	depth := recursionDepth(parentCtx)
	if t.maxRecursion > 0 && depth >= t.maxRecursion {
		return errResult(fmt.Sprintf("recursion depth %d reached max_recursion=%d", depth, t.maxRecursion)), nil
	}
	parentCtx = withRecursion(parentCtx, depth+1)

	if t.timeout > 0 {
		var cancel context.CancelFunc
		parentCtx, cancel = context.WithTimeout(parentCtx, t.timeout)
		defer cancel()
	}

	specRaw, ok := margs["spec"]
	if !ok {
		return errResult("missing required argument \"spec\""), nil
	}
	specBytes, err := json.Marshal(specRaw)
	if err != nil {
		return errResult(fmt.Sprintf("spec marshal: %v", err)), nil
	}
	var spec Spec
	if err := json.Unmarshal(specBytes, &spec); err != nil {
		return errResult(fmt.Sprintf("spec invalid: %v", err)), nil
	}
	AssignPaths(&spec, "")

	if err := t.validate(&spec); err != nil {
		return errResult(err.Error()), nil
	}

	initialInput := ""
	if v, ok := margs["initial_input"].(string); ok {
		initialInput = v
	}

	stateSeed := map[string]any{}
	if t.inheritState {
		if st := toolCtx.State(); st != nil {
			for k, v := range st.All() {
				if strings.HasPrefix(k, "_adk") {
					continue
				}
				stateSeed[k] = v
			}
		}
	}

	exec := &executor{
		tool:      t,
		toolCtx:   toolCtx,
		parentCtx: parentCtx,
		stateSeed: stateSeed,
		outputs:   newOutputs(),
	}

	final, err := exec.run(parentCtx, &spec, initialInput)

	result := map[string]any{
		"outputs":      exec.outputs.snapshot(),
		"final_output": final,
	}
	if err != nil {
		result["error"] = err.Error()
		if final == "" {
			result["final_output"] = exec.outputs.finalOutput()
		}
	}
	return result, nil
}

// ProcessRequest implements the internal RequestProcessor hook so the tool
// declaration is packed into LLM requests.
func (t *flowTool) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return toolutils.PackTool(req, t)
}

func errResult(msg string) map[string]any {
	return map[string]any{
		"outputs":      map[string]any{},
		"final_output": "",
		"error":        msg,
	}
}
