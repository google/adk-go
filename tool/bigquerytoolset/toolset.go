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
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
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
)

// New returns a toolset with the BigQuery metadata tools: list_dataset_ids
// and get_dataset_info.
//
// New creates the HTTP client the tools share, so it fails when cfg is invalid
// or, with the default credentials, when no Application Default Credentials
// are available. The client ignores the cancellation of ctx but keeps its
// values for fetching credentials later. The toolset is safe for concurrent
// use.
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
		add(newTool(b, "list_dataset_ids", listDatasetIDsDescription, b.listDatasetIDs)),
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
	b.applyDefaults(schema)

	return functiontool.New(functiontool.Config{
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
}

// applyDefaults makes each argument with a configured default optional and
// says so in its description, as adk-python's BigQueryTool does. Without
// this the model cannot know which project or dataset the agent builder
// meant, and asks the user at the start of every conversation.
func (b *bigQuery) applyDefaults(schema *jsonschema.Schema) {
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
	}
}
