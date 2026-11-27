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
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
)

// The patterns below are adk-python's. The model writes the table, column
// and model names and the filter that go into the search query, so they are
// held to a grammar that cannot express anything but a name or a simple
// condition.
var (
	identifierPart = "(?:[A-Za-z_][A-Za-z0-9_]*|`[^`\\\\]+`|\"[^\"\\\\]+\")"
	identifier     = identifierPart + `(?:\.` + identifierPart + `)*`

	safeIdentifierRE = regexp.MustCompile(`^` + identifier + `$`)
	safeFilterRE     = regexp.MustCompile(safeFilterPattern())

	vertexAIEndpointRE = regexp.MustCompile(`^projects/[\w-]+/locations/[\w-]+/publishers/[\w-]+/models/[\w.-]+$`)
)

func safeFilterPattern() string {
	const (
		operators   = `(?:=|!=|<=|>=|<|>|(?i:\bLIKE\b|\bIS\s+NOT\b|\bIS\b))`
		value       = `(?:[+-]?\d+(?:\.\d+)?|'[^'\\]*'|(?i:\bTRUE\b|\bFALSE\b|\bNULL\b))`
		inOperator  = `(?i:\bNOT\s+IN\b|\bIN\b)`
		inValues    = `\(\s*` + value + `(?:\s*,\s*` + value + `)*\s*\)`
		between     = `(?i:\bBETWEEN\b)`
		betweenArgs = value + `\s+(?i:\bAND\b)\s+` + value
		join        = `\s+(?i:\bAND\b|\bOR\b)\s+`
	)
	base := `(?:` +
		identifier + `\s*` + operators + `\s*` + value +
		`|` + identifier + `\s*` + inOperator + `\s*` + inValues +
		`|` + identifier + `\s*` + between + `\s*` + betweenArgs +
		`|` + identifier +
		`|1\s*=\s*1)`
	block0 := base + `(?:` + join + base + `)*`
	cond1 := `(?:` + base + `|\(\s*` + block0 + `\s*\))`
	block1 := cond1 + `(?:` + join + cond1 + `)*`
	cond2 := `(?:` + base + `|\(\s*` + block1 + `\s*\))`
	return `(?i)^\s*` + cond2 + `(?:` + join + cond2 + `)*\s*$`
}

func validateIdentifier(value, arg string) error {
	if !safeIdentifierRE.MatchString(strings.TrimSpace(value)) {
		return fmt.Errorf("invalid SQL identifier for %s: identifiers must contain only alphanumeric characters, underscores, and dots, or be quoted with backticks or double quotes", arg)
	}
	return nil
}

func validateFilter(filter string) error {
	if !safeFilterRE.MatchString(filter) {
		return errors.New("additional_filter contains unsafe or unsupported patterns: only simple filters using =, !=, <=, >=, <, >, LIKE, IS, IS NOT, IN, BETWEEN joined by AND or OR (with up to 2 levels of nested parentheses) are allowed")
	}
	return nil
}

type embeddingOptions struct {
	VertexAIEmbeddingModelName                      string `json:"vertex_ai_embedding_model_name,omitempty"`
	SpannerGoogleSQLEmbeddingModelName              string `json:"spanner_googlesql_embedding_model_name,omitempty"`
	SpannerPostgreSQLVertexAIEmbeddingModelEndpoint string `json:"spanner_postgresql_vertex_ai_embedding_model_endpoint,omitempty"`
	OutputDimensionality                            *int   `json:"output_dimensionality,omitempty"`
}

type searchOptions struct {
	TopK                      *int   `json:"top_k,omitempty"`
	DistanceType              string `json:"distance_type,omitempty"`
	NearestNeighborsAlgorithm string `json:"nearest_neighbors_algorithm,omitempty"`
	NumLeavesToSearch         *int   `json:"num_leaves_to_search,omitempty"`
}

type similaritySearchArgs struct {
	databaseArgs
	TableName               string           `json:"table_name"`
	Query                   string           `json:"query"`
	EmbeddingColumnToSearch string           `json:"embedding_column_to_search"`
	Columns                 []string         `json:"columns"`
	EmbeddingOptions        embeddingOptions `json:"embedding_options"`
	AdditionalFilter        string           `json:"additional_filter,omitempty"`
	SearchOptions           *searchOptions   `json:"search_options,omitempty"`
}

const (
	defaultTopK              = 4
	defaultNumLeavesToSearch = 1000
)

func (s *spanner) similaritySearch(ctx agent.Context, args similaritySearchArgs) (any, error) {
	if err := validateIdentifier(args.TableName, "table_name"); err != nil {
		return nil, err
	}
	if err := validateIdentifier(args.EmbeddingColumnToSearch, "embedding_column_to_search"); err != nil {
		return nil, err
	}
	for _, c := range args.Columns {
		if err := validateIdentifier(c, "columns"); err != nil {
			return nil, err
		}
	}
	if args.AdditionalFilter != "" {
		if err := validateFilter(args.AdditionalFilter); err != nil {
			return nil, err
		}
	}
	opts := args.EmbeddingOptions
	if opts.SpannerGoogleSQLEmbeddingModelName != "" {
		if err := validateIdentifier(opts.SpannerGoogleSQLEmbeddingModelName, "spanner_googlesql_embedding_model_name"); err != nil {
			return nil, err
		}
	}
	if opts.SpannerPostgreSQLVertexAIEmbeddingModelEndpoint != "" && !vertexAIEndpointRE.MatchString(opts.SpannerPostgreSQLVertexAIEmbeddingModelEndpoint) {
		return nil, errors.New("invalid Vertex AI endpoint format: expected projects/$project/locations/$location/publishers/google/models/$model")
	}
	set := 0
	for _, name := range []string{opts.VertexAIEmbeddingModelName, opts.SpannerGoogleSQLEmbeddingModelName, opts.SpannerPostgreSQLVertexAIEmbeddingModelEndpoint} {
		if name != "" {
			set++
		}
	}
	if set != 1 {
		return nil, errors.New("exactly one embedding model option must be specified")
	}
	if opts.OutputDimensionality != nil && opts.SpannerGoogleSQLEmbeddingModelName != "" {
		// ML.PREDICT does not take an output dimensionality for embedding
		// models.
		return nil, errors.New("embedding_options[output_dimensionality] is not supported when embedding_options['spanner_googlesql_embedding_model_name'] is specified")
	}

	req := searchRequest{
		databaseArgs:      args.databaseArgs,
		table:             args.TableName,
		embeddingColumn:   args.EmbeddingColumnToSearch,
		columns:           args.Columns,
		filter:            args.AdditionalFilter,
		query:             args.Query,
		embedding:         opts,
		topK:              defaultTopK,
		distanceType:      DistanceCosine,
		algorithm:         ExactNearestNeighbors,
		numLeavesToSearch: defaultNumLeavesToSearch,
	}
	if so := args.SearchOptions; so != nil {
		if so.TopK != nil {
			req.topK = *so.TopK
		}
		req.distanceType = cmp.Or(so.DistanceType, req.distanceType)
		req.algorithm = cmp.Or(so.NearestNeighborsAlgorithm, req.algorithm)
		if so.NumLeavesToSearch != nil {
			req.numLeavesToSearch = *so.NumLeavesToSearch
		}
	}
	return s.search(ctx, req)
}

type vectorStoreSimilaritySearchArgs struct {
	Query string `json:"query"`
}

// vectorStoreSimilaritySearch searches the table [Config.VectorStore] names.
// As in adk-python, its names and filter are not validated: they come from the
// agent's code, not from the model.
func (s *spanner) vectorStoreSimilaritySearch(ctx agent.Context, args vectorStoreSimilaritySearchArgs) (any, error) {
	vs := s.vectorStore
	columns := vs.SelectedColumns
	if len(columns) == 0 {
		columns = []string{vs.ContentColumn}
	}
	return s.search(ctx, searchRequest{
		databaseArgs:    databaseArgs{ProjectID: vs.ProjectID, InstanceID: vs.InstanceID, DatabaseID: vs.DatabaseID},
		table:           vs.TableName,
		embeddingColumn: vs.EmbeddingColumn,
		columns:         columns,
		filter:          vs.AdditionalFilter,
		query:           args.Query,
		embedding: embeddingOptions{
			VertexAIEmbeddingModelName: vs.EmbeddingModel,
			OutputDimensionality:       &vs.VectorLength,
		},
		topK:              cmp.Or(vs.TopK, defaultTopK),
		distanceType:      cmp.Or(vs.DistanceType, DistanceCosine),
		algorithm:         cmp.Or(vs.NearestNeighborsAlgorithm, ExactNearestNeighbors),
		numLeavesToSearch: cmp.Or(vs.NumLeavesToSearch, defaultNumLeavesToSearch),
	})
}

// searchRequest is a similarity search with its options resolved.
type searchRequest struct {
	databaseArgs
	table             string
	embeddingColumn   string
	columns           []string
	filter            string
	query             string
	embedding         embeddingOptions
	topK              int
	distanceType      string
	algorithm         string
	numLeavesToSearch int
}

func (s *spanner) search(ctx agent.Context, req searchRequest) (any, error) {
	if req.algorithm != ExactNearestNeighbors && req.algorithm != ApproximateNearestNeighbors {
		return nil, fmt.Errorf("unsupported search_options['nearest_neighbors_algorithm']: %s", req.algorithm)
	}
	db, err := req.path()
	if err != nil {
		return nil, err
	}
	dialect, err := s.dialect(ctx, db)
	if err != nil {
		return nil, err
	}
	pg := dialect == dialectPostgreSQL
	opts := req.embedding
	if !pg && opts.VertexAIEmbeddingModelName == "" && opts.SpannerGoogleSQLEmbeddingModelName == "" {
		return nil, errors.New("embedding_options['vertex_ai_embedding_model_name'] or embedding_options['spanner_googlesql_embedding_model_name'] must be specified for GoogleSQL dialect Spanner database")
	}
	if pg && opts.VertexAIEmbeddingModelName == "" && opts.SpannerPostgreSQLVertexAIEmbeddingModelEndpoint == "" {
		return nil, errors.New("embedding_options['vertex_ai_embedding_model_name'] or embedding_options['spanner_postgresql_vertex_ai_embedding_model_endpoint'] must be specified for PostgreSQL dialect Spanner database")
	}

	var rows []any
	err = s.withSession(ctx, db, "", func(ss *dbSession) error {
		embedding, err := s.embed(ctx, ss, pg, req)
		if err != nil {
			return err
		}
		sql, err := searchSQL(pg, req)
		if err != nil {
			return err
		}
		param := "embedding"
		if pg {
			param = "p1"
		}
		rs, err := ss.query(ctx, statement{SQL: sql, Params: map[string]any{param: embedding}}, singleUseReadOnly, -1)
		if err != nil {
			return err
		}
		rows = make([]any, len(rs.rows))
		for i, row := range rs.rows {
			rows[i] = row
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return success("rows", rows), nil
}

// embed returns the embedding of the search query: from Vertex AI, or from
// Spanner's ML.PREDICT or spanner.ML_PREDICT_ROW for a model Spanner calls.
func (s *spanner) embed(ctx agent.Context, ss *dbSession, pg bool, req searchRequest) ([]float64, error) {
	opts := req.embedding
	if opts.VertexAIEmbeddingModelName != "" {
		return embedWithVertexAI(ctx, opts.VertexAIEmbeddingModelName, req.query, opts.OutputDimensionality)
	}
	var stmt statement
	if pg {
		stmt = statement{SQL: postgreSQLEmbeddingQuery(opts.SpannerPostgreSQLVertexAIEmbeddingModelEndpoint, opts.OutputDimensionality), Params: map[string]any{"p1": req.query}}
	} else {
		stmt = statement{SQL: googleSQLEmbeddingQuery(opts.SpannerGoogleSQLEmbeddingModelName), Params: map[string]any{"query": req.query}}
	}
	rs, err := ss.query(ctx, stmt, singleUseReadOnly, -1)
	if err != nil {
		return nil, err
	}
	if len(rs.rows) != 1 || len(rs.rows[0]) != 1 {
		return nil, errors.New("the embedding query did not return one embedding")
	}
	values, _ := rs.rows[0][0].([]any)
	embedding := make([]float64, len(values))
	for i, v := range values {
		f, ok := v.(float64)
		if !ok {
			return nil, errors.New("the embedding query returned a value that is not a number")
		}
		embedding[i] = f
	}
	return embedding, nil
}

func embedWithVertexAI(ctx agent.Context, model, query string, outputDimensionality *int) ([]float64, error) {
	client, err := newGenAIClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to embed content: %w", err)
	}
	cfg := &genai.EmbedContentConfig{}
	if outputDimensionality != nil {
		d := *outputDimensionality
		if d <= 0 || d > math.MaxInt32 {
			return nil, errors.New("embedding_options[output_dimensionality] must be a positive 32-bit integer")
		}
		cfg.OutputDimensionality = new(int32(d))
	}
	resp, err := client.Models.EmbedContent(ctx, model, genai.Text(query), cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to embed content: %w", err)
	}
	if len(resp.Embeddings) == 0 || resp.Embeddings[0] == nil {
		return nil, errors.New("failed to embed content: the model returned no embedding")
	}
	values := resp.Embeddings[0].Values
	embedding := make([]float64, len(values))
	for i, v := range values {
		embedding[i] = float64(v)
	}
	return embedding, nil
}

func googleSQLEmbeddingQuery(model string) string {
	return `
    SELECT embeddings.values
    FROM ML.PREDICT(
      MODEL ` + model + `,
      (SELECT CAST(@query AS STRING) as content)
    )
  `
}

func postgreSQLEmbeddingQuery(endpoint string, outputDimensionality *int) string {
	args := `
      'instances',
      JSONB_BUILD_ARRAY(
          JSONB_BUILD_OBJECT(
              'content',
              $1::TEXT
          )
      )
  `
	if outputDimensionality != nil {
		args += `,
        'parameters',
        JSONB_BUILD_OBJECT(
            'outputDimensionality',
            ` + strconv.Itoa(*outputDimensionality) + `
        )
    `
	}
	return `
      SELECT spanner.FLOAT32_ARRAY(
          spanner.ML_PREDICT_ROW(
              '` + endpoint + `',
              JSONB_BUILD_OBJECT(
                  ` + args + `
              )
          ) -> 'predictions' -> 0 -> 'embeddings' -> 'values'
      )
  `
}

// distanceFunction returns the SQL function that measures distanceType: in
// PostgreSQL when pg is set, else in GoogleSQL, approximate when ann is set.
func distanceFunction(distanceType string, ann, pg bool) (string, error) {
	var functions map[string]string
	switch {
	case pg:
		functions = map[string]string{DistanceCosine: "spanner.cosine_distance", DistanceEuclidean: "spanner.euclidean_distance", DistanceDotProduct: "spanner.dot_product"}
	case ann:
		functions = map[string]string{DistanceCosine: "APPROX_COSINE_DISTANCE", DistanceEuclidean: "APPROX_EUCLIDEAN_DISTANCE", DistanceDotProduct: "APPROX_DOT_PRODUCT"}
	default:
		functions = map[string]string{DistanceCosine: "COSINE_DISTANCE", DistanceEuclidean: "EUCLIDEAN_DISTANCE", DistanceDotProduct: "DOT_PRODUCT"}
	}
	f, ok := functions[distanceType]
	if !ok {
		return "", fmt.Errorf("unsupported distance type %q: use %s, %s or %s", distanceType, DistanceCosine, DistanceEuclidean, DistanceDotProduct)
	}
	return f, nil
}

// searchSQL is adk-python's kNN or ANN query for req, which orders the rows
// by their distance from the query embedding, passed as a parameter.
func searchSQL(pg bool, req searchRequest) (string, error) {
	ann := req.algorithm == ApproximateNearestNeighbors
	if ann && pg {
		return "", fmt.Errorf("%s is not supported for PostgreSQL dialect", ApproximateNearestNeighbors)
	}
	fn, err := distanceFunction(req.distanceType, ann, pg)
	if err != nil {
		return "", err
	}
	param := "@embedding"
	if pg {
		param = "$1"
	}
	distance := fn + "(\n      " + req.embeddingColumn + ",\n      " + param
	if ann {
		distance += `,
      options => JSON '{"num_leaves_to_search": ` + strconv.Itoa(req.numLeavesToSearch) + `}'`
	}
	distance += ") AS distance\n  "
	selected := strings.Join(append(append([]string{}, req.columns...), distance), ", ")

	var where, limit string
	if ann {
		where = req.embeddingColumn + " IS NOT NULL"
		if req.filter != "" {
			where += " AND " + req.filter
		}
		limit = "LIMIT " + strconv.Itoa(req.topK)
	} else {
		where = cmp.Or(req.filter, "1=1")
		if req.topK > 0 {
			limit = "LIMIT " + strconv.Itoa(req.topK)
		}
	}
	return `
    SELECT ` + selected + `
    FROM ` + req.table + `
    WHERE ` + where + `
    ORDER BY distance
    ` + limit + `
  `, nil
}
