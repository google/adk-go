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

package geminitool

import (
	"fmt"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
)

// VertexAiSearch is a built-in tool that uses Vertex AI Search to retrieve
// information from configured data stores or search engines.
// The tool operates internally within the model and does not require or
// perform local code execution.
//
// Either DataStoreID or SearchEngineID must be specified (but not both).
// If DataStoreSpecs is provided, SearchEngineID must also be provided.
type VertexAiSearch struct {
	// DataStoreID is the Vertex AI search data store resource ID in the format:
	// `projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}`
	DataStoreID string

	// DataStoreSpecs are specifications that define the specific DataStores to be searched.
	// This should only be set if SearchEngineID is also set.
	DataStoreSpecs []*genai.VertexAISearchDataStoreSpec

	// SearchEngineID is the Vertex AI search engine resource ID in the format:
	// `projects/{project}/locations/{location}/collections/{collection}/engines/{engine}`
	SearchEngineID string

	// Filter is an optional filter to apply to the search results.
	Filter string

	// MaxResults is the maximum number of results to return.
	MaxResults *int32
}

// Name implements tool.Tool.
func (v *VertexAiSearch) Name() string {
	return "vertex_ai_search"
}

// Description implements tool.Tool.
func (v *VertexAiSearch) Description() string {
	return "Retrieves information from Vertex AI Search data stores or search engines."
}

// ProcessRequest adds the VertexAiSearch tool to the LLM request.
func (v *VertexAiSearch) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	if (v.DataStoreID == "" && v.SearchEngineID == "") ||
		(v.DataStoreID != "" && v.SearchEngineID != "") {
		return fmt.Errorf("either DataStoreID or SearchEngineID must be specified (but not both)")
	}

	if len(v.DataStoreSpecs) > 0 && v.SearchEngineID == "" {
		return fmt.Errorf("SearchEngineID must be specified if DataStoreSpecs is provided")
	}

	vertexAISearch := &genai.VertexAISearch{
		Filter:     v.Filter,
		MaxResults: v.MaxResults,
	}

	if v.DataStoreID != "" {
		vertexAISearch.Datastore = v.DataStoreID
	}
	if v.SearchEngineID != "" {
		vertexAISearch.Engine = v.SearchEngineID
		vertexAISearch.DataStoreSpecs = v.DataStoreSpecs
	}

	return setTool(req, &genai.Tool{
		Retrieval: &genai.Retrieval{
			VertexAISearch: vertexAISearch,
		},
	})
}

// IsLongRunning implements tool.Tool.
func (v *VertexAiSearch) IsLongRunning() bool {
	return false
}
