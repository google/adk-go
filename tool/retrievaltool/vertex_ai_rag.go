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

package retrievaltool

import (
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// VertexAIRAG is a retrieval tool that uses Vertex AI RAG (Retrieval-Augmented Generation) to retrieve data.
type VertexAIRAG struct {
	name           string
	description    string
	vertexRAGStore *genai.VertexRAGStore
}

// NewVertexAIRAG creates a new Vertex AI RAG retrieval tool with the given parameters.
func NewVertexAIRAG(name, description string, vertexRAGStore *genai.VertexRAGStore) (tool.Tool, error) {
	return &VertexAIRAG{
		name:           name,
		description:    description,
		vertexRAGStore: vertexRAGStore,
	}, nil
}

// Name implements tool.Tool.
func (v *VertexAIRAG) Name() string {
	return v.name
}

// Description implements tool.Tool.
func (v *VertexAIRAG) Description() string {
	return v.description
}

// IsLongRunning implements tool.Tool.
func (v *VertexAIRAG) IsLongRunning() bool {
	return false
}

// ProcessRequest adds the Vertex AI RAG tool to the LLM request.
// Uses the built-in Vertex AI RAG tool for Gemini models.
func (v *VertexAIRAG) ProcessRequest(ctx tool.Context, req *model.LLMRequest) error {
	return v.addBuiltInRAGTool(req)
}

// addBuiltInRAGTool adds the built-in Vertex AI RAG tool to the request config.
func (v *VertexAIRAG) addBuiltInRAGTool(req *model.LLMRequest) error {
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}

	ragTool := &genai.Tool{
		Retrieval: &genai.Retrieval{
			VertexRAGStore: v.vertexRAGStore,
		},
	}

	req.Config.Tools = append(req.Config.Tools, ragTool)
	return nil
}
