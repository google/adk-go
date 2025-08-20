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

package tool

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"google.golang.org/adk/llm"
	"google.golang.org/genai"
)

func NewGenaiTool(t *genai.Tool) (GenaiTool, error) {
	b, err := json.Marshal(t)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tool: %w", err)
	}

	m := make(map[string]any)
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("failed to unmarshal encoded tool: %w", err)
	}

	if len(m) != 1 {
		return nil, fmt.Errorf("genai tool should have only 1 field, got: %v", m)
	}

	name := slices.Collect(maps.Keys(m))[0]
	description := name

	return &genaiTool{
		name:        name,
		description: description,
		value:       t,
	}, nil
}

func MustNewGenaiTool(t *genai.Tool) GenaiTool {
	res, err := NewGenaiTool(t)
	if err != nil {
		panic(err)
	}
	return res
}

type genaiTool struct {
	name, description string
	value             *genai.Tool
}

func (t *genaiTool) Value() *genai.Tool {
	return t.value
}

func (t *genaiTool) Name() string {
	return t.name
}

func (t *genaiTool) Description() string {
	return t.description
}

func (t *genaiTool) ProcessRequest(ctx Context, req *llm.Request) error {
	if req.GenerateConfig == nil {
		req.GenerateConfig = &genai.GenerateContentConfig{}
	}

	req.GenerateConfig.Tools = append(req.GenerateConfig.Tools, t.Value())
	return nil
}
