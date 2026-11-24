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

	"google.golang.org/api/option"
)

// Config configures the Spanner toolset. The zero value is valid: the tools
// authenticate with Application Default Credentials and the model names the
// project, instance and database in every call.
type Config struct {
	// ClientOptions configure the connection to the Spanner API, for example
	// [option.WithAuthCredentialsFile], [option.WithTokenSource] or
	// [option.WithEndpoint]. Without credentials among them, the tools use
	// Application Default Credentials with the spanner.admin and spanner.data
	// scopes. The tools read a database's dialect through the database admin
	// API, so the credentials need spanner.databases.get as well as permission
	// to query the database.
	ClientOptions []option.ClientOption

	// ExternalAccessTokenKey, when set, is the session state key holding the
	// OAuth access token each call authenticates with, for an application that
	// obtains the end user's token itself. A call fails, and the model is told
	// so, when the state has no non-empty string under the key. New rejects
	// [option.WithAPIKey], [option.WithTokenSource], [option.WithCredentials],
	// [option.WithAuthCredentials], [option.WithAuthCredentialsFile] and
	// [option.WithAuthCredentialsJSON] alongside it. It cannot see inside an
	// [option.WithHTTPClient]: such a client must not authenticate requests
	// itself, or its credentials replace the end user's.
	ExternalAccessTokenKey string

	// MaxQueryResultRows is the most rows spanner_execute_sql returns from a
	// query. Zero means 50, adk-python's default. When rows are left out, the
	// result has result_is_likely_truncated set to true. adk-python also sets
	// it when a query returns exactly this many rows.
	MaxQueryResultRows int

	// QueryResultMode is the shape of the rows spanner_execute_sql returns.
	QueryResultMode QueryResultMode

	// DatabaseRole, when set, is the database role spanner_execute_sql
	// queries as, for databases that use fine-grained access control. The
	// other tools query as the credentials' own identity, as in adk-python.
	DatabaseRole string

	// VectorStore, when set, adds the spanner_vector_store_similarity_search
	// tool, which searches this table with only a text query from the model.
	VectorStore *VectorStoreConfig
}

// QueryResultMode is the shape of the rows spanner_execute_sql returns.
type QueryResultMode string

const (
	// QueryResultModeDefault returns each row as a list of values, in column
	// order. It is the zero value.
	QueryResultModeDefault QueryResultMode = ""

	// QueryResultModeDictList returns each row as an object from column name
	// to value. A column without a name, such as COUNT(*), has the key "",
	// and of two columns with the same name only the last is kept.
	QueryResultModeDictList QueryResultMode = "dict_list"
)

// Distance types for vector similarity search.
const (
	DistanceCosine     = "COSINE"
	DistanceEuclidean  = "EUCLIDEAN"
	DistanceDotProduct = "DOT_PRODUCT"
)

// Nearest neighbors algorithms for vector similarity search.
const (
	// ExactNearestNeighbors compares the query with every row (kNN).
	ExactNearestNeighbors = "EXACT_NEAREST_NEIGHBORS"

	// ApproximateNearestNeighbors walks a vector index (ANN). It needs a
	// vector index on the embedding column and a GoogleSQL dialect database.
	ApproximateNearestNeighbors = "APPROXIMATE_NEAREST_NEIGHBORS"
)

// VectorStoreConfig names a Spanner table that holds text contents and their
// embeddings, which spanner_vector_store_similarity_search searches. The
// table names and filter are trusted: they go into the query as written.
type VectorStoreConfig struct {
	// ProjectID, InstanceID and DatabaseID name the database holding the
	// table. They are required.
	ProjectID  string
	InstanceID string
	DatabaseID string

	// TableName is the vector store table. It is required.
	TableName string

	// ContentColumn holds the text of each row. It is required, and returned
	// in the search results when SelectedColumns is empty.
	ContentColumn string

	// EmbeddingColumn holds the embedding of each row's content. It is
	// required.
	EmbeddingColumn string

	// VectorLength is the dimension of the embeddings in EmbeddingColumn. It
	// must be positive, and the search asks the embedding model for vectors
	// of this length.
	VectorLength int

	// EmbeddingModel is the Vertex AI embedding model the query is embedded
	// with, such as "text-embedding-005". It is required. The embedding call
	// uses a genai client configured from the environment, as adk-python's
	// genai.Client() is: set GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT
	// and GOOGLE_CLOUD_LOCATION to call Vertex AI.
	EmbeddingModel string

	// SelectedColumns are the columns each search result holds, before the
	// distance. Empty means ContentColumn alone.
	SelectedColumns []string

	// NearestNeighborsAlgorithm is ExactNearestNeighbors, the default when
	// empty, or ApproximateNearestNeighbors.
	NearestNeighborsAlgorithm string

	// TopK is the number of rows a search returns. Zero means 4.
	TopK int

	// DistanceType is DistanceCosine, the default when empty,
	// DistanceEuclidean or DistanceDotProduct.
	DistanceType string

	// NumLeavesToSearch is how many leaves of the vector index an
	// approximate search walks. Zero means 1000. It is ignored for an exact
	// search.
	NumLeavesToSearch int

	// AdditionalFilter, when set, is a condition added to the WHERE clause of
	// the search query.
	AdditionalFilter string
}

const defaultMaxQueryResultRows = 50

func (c Config) validate() error {
	if c.MaxQueryResultRows < 0 {
		return errors.New("spannertoolset: Config.MaxQueryResultRows must not be negative")
	}
	switch c.QueryResultMode {
	case QueryResultModeDefault, QueryResultModeDictList:
	default:
		return fmt.Errorf("spannertoolset: Config.QueryResultMode %q is not a known mode", c.QueryResultMode)
	}
	if c.VectorStore != nil {
		if err := c.VectorStore.validate(); err != nil {
			return fmt.Errorf("spannertoolset: Config.VectorStore: %w", err)
		}
	}
	return nil
}

func (v *VectorStoreConfig) validate() error {
	for _, f := range []struct{ name, value string }{
		{"ProjectID", v.ProjectID},
		{"InstanceID", v.InstanceID},
		{"DatabaseID", v.DatabaseID},
		{"TableName", v.TableName},
		{"ContentColumn", v.ContentColumn},
		{"EmbeddingColumn", v.EmbeddingColumn},
		{"EmbeddingModel", v.EmbeddingModel},
	} {
		if f.value == "" {
			return fmt.Errorf("%s is required", f.name)
		}
	}
	if v.VectorLength <= 0 {
		return errors.New("VectorLength must be positive")
	}
	if v.TopK < 0 {
		return errors.New("TopK must not be negative")
	}
	if v.NumLeavesToSearch < 0 {
		return errors.New("NumLeavesToSearch must not be negative")
	}
	if _, err := distanceFunction(cmp.Or(v.DistanceType, DistanceCosine), false, false); err != nil {
		return err
	}
	switch v.NearestNeighborsAlgorithm {
	case "", ExactNearestNeighbors, ApproximateNearestNeighbors:
	default:
		return fmt.Errorf("NearestNeighborsAlgorithm %q is not %s or %s", v.NearestNeighborsAlgorithm, ExactNearestNeighbors, ApproximateNearestNeighbors)
	}
	return nil
}
