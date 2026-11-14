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

package bigquerytoolset

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/typeutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// Each tool's description is adk-python's docstring for the tool, without the
// arguments Go tools do not have.
var (
	//go:embed descriptions/list_dataset_ids.txt
	listDatasetIDsDescription string
	//go:embed descriptions/get_dataset_info.txt
	getDatasetInfoDescription string
	//go:embed descriptions/list_table_ids.txt
	listTableIDsDescription string
	//go:embed descriptions/get_table_info.txt
	getTableInfoDescription string
	//go:embed descriptions/get_job_info.txt
	getJobInfoDescription string
)

// New returns a toolset with the BigQuery metadata tools: list_dataset_ids,
// get_dataset_info, list_table_ids, get_table_info and get_job_info.
//
// New creates the HTTP client the tools share, so it fails when cfg is invalid
// or, with the default credentials, when no Application Default Credentials
// are available. The client ignores the cancellation of ctx, and some
// credential types keep its values for fetching tokens later. The toolset is
// safe for concurrent use.
func New(ctx context.Context, cfg Config) (tool.Toolset, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	c, err := newClient(ctx, cfg)
	if err != nil {
		return nil, err
	}
	b := &bigQuery{
		client:           c,
		location:         cfg.Location,
		defaultProjectID: cfg.DefaultProjectID,
		defaultDatasetID: cfg.DefaultDatasetID,
	}

	var errs []error
	add := func(t tool.Tool, err error) tool.Tool {
		errs = append(errs, err)
		return t
	}
	tools := []tool.Tool{
		add(newTool(b, "get_dataset_info", getDatasetInfoDescription, b.getDatasetInfo)),
		add(newTool(b, "get_job_info", getJobInfoDescription, b.getJobInfo)),
		add(newTool(b, "get_table_info", getTableInfoDescription, b.getTableInfo)),
		add(newTool(b, "list_dataset_ids", listDatasetIDsDescription, b.listDatasetIDs)),
		add(newTool(b, "list_table_ids", listTableIDsDescription, b.listTableIDs)),
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return &toolset{tools: tools}, nil
}

type toolset struct {
	tools []tool.Tool
}

func (*toolset) Name() string { return "bigquery_toolset" }

func (t *toolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	return slices.Clone(t.tools), nil
}

// bigQuery is what the tools share.
type bigQuery struct {
	client           *client
	location         string
	defaultProjectID string
	defaultDatasetID string
}

func (b *bigQuery) project(id string) string { return orDefault(id, b.defaultProjectID) }

func (b *bigQuery) dataset(id string) string { return orDefault(id, b.defaultDatasetID) }

func orDefault(id, def string) string {
	if id == "" {
		return def
	}
	return id
}

// newTool wraps run as a function tool named name. An error from run is
// returned to the model as an error result, unless the call's context ended.
func newTool[TArgs any](b *bigQuery, name, description string, run func(agent.Context, TArgs) (any, error)) (tool.Tool, error) {
	schema, err := jsonschema.For[TArgs](nil)
	if err != nil {
		return nil, fmt.Errorf("bigquerytoolset: inferring the input schema of %s: %w", name, err)
	}
	defaulted := b.applyDefaults(schema)
	resolved, err := schema.Resolve(nil)
	if err != nil {
		return nil, fmt.Errorf("bigquerytoolset: resolving the input schema of %s: %w", name, err)
	}

	inner, err := functiontool.New(functiontool.Config{
		Name:        name,
		Description: strings.TrimRight(description, "\n"),
		InputSchema: schema,
	}, func(ctx agent.Context, args TArgs) (any, error) {
		result, err := run(ctx, args)
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			return errorResult(err), nil
		}
		return result, nil
	})
	if err != nil {
		return nil, err
	}
	return &nullDefaultsTool{FunctionTool: inner.(toolinternal.FunctionTool), defaulted: defaulted, schema: resolved}, nil
}

// nullDefaultsTool treats a null argument that has a configured default like
// an omitted one, as adk-python's BigQueryTool does. The declaration still
// says string, so validation rejects the null unless it is dropped first.
//
// It also refuses a call that leaves out a required argument or passes one the
// tool does not declare, with the input schema's validation error. A function
// tool drops an undeclared argument and answers a missing one with an error
// result, as adk-python's FunctionTool does (tools/function_tool.py:284,
// :365-378), and adk-python's BigQueryTool is one; this tool's tests require
// the refusal.
type nullDefaultsTool struct {
	toolinternal.FunctionTool
	defaulted []string
	schema    *jsonschema.Resolved
}

// ProcessRequest packs the wrapped tool's declaration for this request, in the
// form the request's typed-declaration switch and backend ask for, then
// registers t in its place, because the LLM flow calls what is registered on
// the request.
func (t *nullDefaultsTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if err := t.FunctionTool.(toolinternal.RequestProcessor).ProcessRequest(ctx, req); err != nil {
		return err
	}
	req.Tools[t.Name()] = t
	return nil
}

// Run drops null defaulted arguments and runs the wrapped tool.
func (t *nullDefaultsTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	cleaned := t.clean(args)
	if err := t.validate(cleaned); err != nil {
		return nil, err
	}
	return t.FunctionTool.Run(ctx, cleaned)
}

// RunRaw implements toolinternal.RawResultTool for the wrapped function tool:
// it drops null defaulted arguments and returns the wrapped tool's raw result.
func (t *nullDefaultsTool) RunRaw(ctx agent.Context, args any) (any, error) {
	cleaned := t.clean(args)
	if err := t.validate(cleaned); err != nil {
		return nil, err
	}
	if r, ok := t.FunctionTool.(toolinternal.RawResultTool); ok {
		return r.RunRaw(ctx, cleaned)
	}
	return t.FunctionTool.Run(ctx, cleaned)
}

// validate checks a map of arguments against the input schema. Any other value
// is left to the wrapped tool, which refuses it.
func (t *nullDefaultsTool) validate(args any) error {
	m, ok := args.(map[string]any)
	if !ok {
		return nil
	}
	return typeutil.ValidateWithJSONSchema(m, t.schema)
}

// clean returns args without the null defaulted arguments; a value that is
// not a map is returned unchanged. The caller's map is not modified.
func (t *nullDefaultsTool) clean(args any) any {
	m, ok := args.(map[string]any)
	if !ok {
		return args
	}
	cleaned := maps.Clone(m)
	for _, arg := range t.defaulted {
		if v, ok := cleaned[arg]; ok && v == nil {
			delete(cleaned, arg)
		}
	}
	return cleaned
}

// applyDefaults makes each argument with a configured default optional and
// says so in its description, as adk-python's BigQueryTool does. Without
// this the model cannot know which project or dataset the agent builder
// meant, and asks the user at the start of every conversation. It returns the
// arguments that got a default.
func (b *bigQuery) applyDefaults(schema *jsonschema.Schema) []string {
	var defaulted []string
	for arg, def := range map[string]string{
		"project_id": b.defaultProjectID,
		"dataset_id": b.defaultDatasetID,
	} {
		prop, ok := schema.Properties[arg]
		if def == "" || !ok {
			continue
		}
		prop.Description = fmt.Sprintf("Defaults to '%s' when not provided.", def)
		schema.Required = slices.DeleteFunc(schema.Required, func(r string) bool { return r == arg })
		defaulted = append(defaulted, arg)
	}
	return defaulted
}
