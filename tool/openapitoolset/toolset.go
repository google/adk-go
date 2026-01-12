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

// Package openapitoolset provides tools generated from OpenAPI specifications.
package openapitoolset

import (
	"encoding/json"
	"fmt"

	"gopkg.in/yaml.v3"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/auth"
	"google.golang.org/adk/tool"
)

// Config provides configuration for the OpenAPI toolset.
type Config struct {
	// SpecDict is the OpenAPI spec as a map. If provided, SpecStr is ignored.
	SpecDict map[string]any
	// SpecStr is the OpenAPI spec as a string (JSON or YAML).
	SpecStr string
	// SpecStrType is the format of SpecStr: "json" or "yaml".
	SpecStrType string
	// AuthScheme defines how the API expects authentication.
	AuthScheme auth.AuthScheme
	// AuthCredential contains the credentials for authentication.
	AuthCredential *auth.AuthCredential
	// ToolFilter selects which tools to include.
	ToolFilter tool.Predicate
	// ToolNamePrefix is prepended to each tool name.
	ToolNamePrefix string
}

// New creates a new OpenAPI toolset from the given configuration.
func New(cfg Config) (tool.Toolset, error) {
	var specDict map[string]any
	if cfg.SpecDict != nil {
		specDict = cfg.SpecDict
	} else if cfg.SpecStr != "" {
		var err error
		specDict, err = loadSpec(cfg.SpecStr, cfg.SpecStrType)
		if err != nil {
			return nil, fmt.Errorf("failed to load OpenAPI spec: %w", err)
		}
	} else {
		return nil, fmt.Errorf("either SpecDict or SpecStr must be provided")
	}

	// Parse the OpenAPI spec into tools
	tools, err := parseOpenAPISpec(specDict)
	if err != nil {
		return nil, fmt.Errorf("failed to parse OpenAPI spec: %w", err)
	}

	// Configure auth on all tools
	for _, t := range tools {
		if cfg.AuthScheme != nil {
			t.authScheme = cfg.AuthScheme
		}
		if cfg.AuthCredential != nil {
			t.authCredential = cfg.AuthCredential
		}
		if cfg.ToolNamePrefix != "" {
			t.name = cfg.ToolNamePrefix + t.name
		}
	}

	return &openAPIToolset{
		tools:      tools,
		toolFilter: cfg.ToolFilter,
	}, nil
}

// loadSpec loads the OpenAPI spec from a string.
func loadSpec(specStr string, specType string) (map[string]any, error) {
	var result map[string]any
	switch specType {
	case "json", "":
		if err := json.Unmarshal([]byte(specStr), &result); err != nil {
			return nil, fmt.Errorf("failed to parse JSON spec: %w", err)
		}
	case "yaml":
		if err := yaml.Unmarshal([]byte(specStr), &result); err != nil {
			return nil, fmt.Errorf("failed to parse YAML spec: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported spec type: %s", specType)
	}
	return result, nil
}

type openAPIToolset struct {
	tools      []*RestApiTool
	toolFilter tool.Predicate
}

// Name implements tool.Toolset.
func (s *openAPIToolset) Name() string {
	return "openapi_toolset"
}

// Tools implements tool.Toolset.
func (s *openAPIToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	var result []tool.Tool
	for _, t := range s.tools {
		if s.toolFilter != nil && !s.toolFilter(ctx, t) {
			continue
		}
		result = append(result, t)
	}
	return result, nil
}

// GetTool returns a specific tool by name.
func (s *openAPIToolset) GetTool(name string) *RestApiTool {
	for _, t := range s.tools {
		if t.name == name {
			return t
		}
	}
	return nil
}
