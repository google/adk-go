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

package spannertoolset

import (
	"cmp"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// Each tool's description is adk-python's docstring for the tool, without the
// arguments Go tools do not have.
var (
	//go:embed descriptions/list_table_names.txt
	listTableNamesDescription string
	//go:embed descriptions/list_table_indexes.txt
	listTableIndexesDescription string
	//go:embed descriptions/list_table_index_columns.txt
	listTableIndexColumnsDescription string
	//go:embed descriptions/list_named_schemas.txt
	listNamedSchemasDescription string
	//go:embed descriptions/get_table_schema.txt
	getTableSchemaDescription string
	//go:embed descriptions/execute_sql.txt
	executeSQLDescription string
	//go:embed descriptions/execute_sql_dict_list.txt
	executeSQLDictListDescription string
	//go:embed descriptions/similarity_search.txt
	similaritySearchDescription string
	//go:embed descriptions/vector_store_similarity_search.txt
	vectorStoreSimilaritySearchDescription string
)

// New returns a toolset with adk-python's SpannerToolset tools:
// spanner_list_table_names, spanner_list_table_indexes,
// spanner_list_table_index_columns, spanner_list_named_schemas,
// spanner_get_table_schema, spanner_execute_sql, which runs read-only
// queries, spanner_similarity_search and, when [Config.VectorStore] is set,
// spanner_vector_store_similarity_search.
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
	s := &spanner{
		client:     c,
		maxRows:    cmp.Or(cfg.MaxQueryResultRows, defaultMaxQueryResultRows),
		resultMode: cfg.QueryResultMode,
		role:       cfg.DatabaseRole,
	}
	if cfg.VectorStore != nil {
		vs := *cfg.VectorStore
		vs.SelectedColumns = slices.Clone(vs.SelectedColumns)
		s.vectorStore = &vs
	}

	executeSQL := executeSQLDescription
	if cfg.QueryResultMode == QueryResultModeDictList {
		executeSQL = executeSQLDictListDescription
	}
	var errs []error
	add := func(t tool.Tool, err error) tool.Tool {
		errs = append(errs, err)
		return t
	}
	tools := []tool.Tool{
		add(newTool("spanner_list_table_names", listTableNamesDescription, s.listTableNames)),
		add(newTool("spanner_list_table_indexes", listTableIndexesDescription, s.listTableIndexes)),
		add(newTool("spanner_list_table_index_columns", listTableIndexColumnsDescription, s.listTableIndexColumns)),
		add(newTool("spanner_list_named_schemas", listNamedSchemasDescription, s.listNamedSchemas)),
		add(newTool("spanner_get_table_schema", getTableSchemaDescription, s.getTableSchema)),
		add(newTool("spanner_execute_sql", executeSQL, s.executeSQL)),
		add(newTool("spanner_similarity_search", similaritySearchDescription, s.similaritySearch)),
	}
	if s.vectorStore != nil {
		tools = append(tools, add(newTool("spanner_vector_store_similarity_search", vectorStoreSimilaritySearchDescription, s.vectorStoreSimilaritySearch)))
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return &toolset{tools: tools}, nil
}

type toolset struct {
	tools []tool.Tool
}

func (*toolset) Name() string { return "spanner_toolset" }

func (t *toolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	return slices.Clone(t.tools), nil
}

// spanner is what the tools share.
type spanner struct {
	client      *client
	maxRows     int
	resultMode  QueryResultMode
	role        string
	vectorStore *VectorStoreConfig
}

// newGenAIClient creates the client that embeds a search query with a Vertex
// AI model. adk-python creates one per search, configured from the
// environment. Tests replace it.
var newGenAIClient = func(ctx context.Context) (*genai.Client, error) {
	return genai.NewClient(ctx, nil)
}

// newTool wraps run as a function tool named name. An error from run is
// returned to the model as an error result, unless the call's context ended.
func newTool[TArgs any](name, description string, run func(agent.Context, TArgs) (any, error)) (tool.Tool, error) {
	schema, err := jsonschema.For[TArgs](nil)
	if err != nil {
		return nil, fmt.Errorf("spannertoolset: inferring the input schema of %s: %w", name, err)
	}
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
			return errorDetails(err.Error()), nil
		}
		return result, nil
	})
}

// errorDetails is what a tool returns to the model when its call fails, in
// adk-python's format.
func errorDetails(details string) map[string]any {
	return map[string]any{
		"status":        "ERROR",
		"error_details": details,
	}
}

// success is what a tool returns to the model when its call succeeds, with
// the result under key, in adk-python's format.
func success(key string, value any) map[string]any {
	return map[string]any{
		"status": "SUCCESS",
		key:      value,
	}
}
