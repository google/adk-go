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

// Package spannertoolset provides a [tool.Toolset] whose tools let an agent
// explore a Spanner database: list its tables, named schemas and indexes,
// read a table's schema, run read-only SQL queries and find the rows whose
// embeddings are closest to a text query.
//
// The tools have the same names, arguments and results as adk-python's
// SpannerToolset, so prompts and skills written for it work unchanged. Every
// query runs in a strong read-only transaction, so spanner_execute_sql cannot
// change data. Like adk-python's, only spanner_list_table_names and the
// similarity searches work on a PostgreSQL dialect database; the other tools
// return an error result for one.
//
// The differences in results: values of types other than BOOL, INT64, ENUM,
// FLOAT64, FLOAT32, JSON, ARRAY and STRUCT, such as TIMESTAMP, DATE, NUMERIC
// and BYTES, are the strings the Spanner REST API sends, where adk-python
// turns a whole row holding one of them into its Python string form; INT64
// values and integers inside JSON values beyond 2^53 in magnitude are decimal
// strings, where adk-python returns numbers; and spanner_execute_sql sets
// result_is_likely_truncated only when rows were left out (see
// [Config.MaxQueryResultRows]). The embedding_options and search_options
// arguments of spanner_similarity_search are declared with their keys, where
// adk-python declares free-form objects, so a value of the wrong type fails
// the call instead of being converted.
//
// When Spanner rejects a call, for example because a table does not exist,
// or an argument is invalid, the tool returns
// {"status": "ERROR", "error_details": "..."} to the model so it can correct
// the call. A canceled or expired context fails the tool call instead.
//
// By default the tools authenticate with Application Default Credentials:
//
//	sp, err := spannertoolset.New(ctx, spannertoolset.Config{})
//	if err != nil {
//		return err
//	}
//	agent, err := llmagent.New(llmagent.Config{
//		// ...
//		Toolsets: []tool.Toolset{sp},
//	})
//
// To expose only some of the tools, wrap the toolset with
// [tool.FilterToolset]. Leaving out spanner_execute_sql and the similarity
// searches gives what adk-python's toolset offers without its DATA_READ
// capability.
package spannertoolset
