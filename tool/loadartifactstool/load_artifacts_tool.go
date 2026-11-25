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

// Package loadartifactstool defines a tool for loading artifacts.
// This tool informs the model about available artifacts and provides their content when
// requested by the model through a function call.
package loadartifactstool

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"slices"

	"golang.org/x/sync/errgroup"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// artifactsTool is a tool that loads artifacts and adds them to the session.
type artifactsTool struct {
	name        string
	description string
}

// New creates a new loadArtifactsTool.
func New() tool.Tool {
	return &artifactsTool{
		name:        "load_artifacts",
		description: "Loads the artifacts and adds them to the session.",
	}
}

// Name implements tool.Tool.
func (t *artifactsTool) Name() string {
	return t.name
}

// Description implements tool.Tool.
func (t *artifactsTool) Description() string {
	return t.description
}

// IsLongRunning implements tool.Tool.
func (t *artifactsTool) IsLongRunning() bool {
	return false
}

// Declaration returns the GenAI FunctionDeclaration for the load_artifacts tool.
//
// This declaration allows the LLM to understand and call the tool
// by specifying the function name, a detailed description of its
// purpose, and the required input parameters (schema).
func (t *artifactsTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        t.name,
		Description: t.description,
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"artifact_names": {
					Type: "ARRAY",
					Items: &genai.Schema{
						Type: "STRING",
					},
				},
			},
		},
	}
}

// Run implements tool.Tool.
func (t *artifactsTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type, got: %T", args)
	}
	var artifactNames []string
	artifactNamesRaw, exists := m["artifact_names"]
	if !exists {
		artifactNames = []string{}
	} else {
		// In order to cast properly from []any to []string we're gonna marshal and then
		// unmarshal the artifact_names value.
		artifactNamesJson, err := json.Marshal(artifactNamesRaw)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal artifact_names to JSON: %w", err)
		}
		// Arguments the model can correct get an error result, not a tool
		// error. A JSON null element would decode to "" without an error, so it
		// is checked separately.
		if err := json.Unmarshal(artifactNamesJson, &artifactNames); err != nil || hasNullElement(artifactNamesRaw) {
			return map[string]any{
				"error":      "'artifact_names' must be a list of strings.",
				"error_code": "INVALID_ARGUMENTS",
			}, nil
		}
		// Ensure the slice is not nil if it's empty
		if artifactNames == nil {
			artifactNames = []string{}
		}
	}
	result := map[string]any{
		"artifact_names": artifactNames,
	}
	return result, nil
}

// hasNullElement reports whether v is a list that holds a JSON null.
func hasNullElement(v any) bool {
	list, _ := v.([]any)
	for _, e := range list {
		if e == nil {
			return true
		}
	}
	return false
}

// ProcessRequest processes the LLM request. It packs the tool, appends initial
// instructions, and processes any load artifacts function calls.
func (t *artifactsTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if ctx.Artifacts() == nil {
		return fmt.Errorf("load_artifacts tool requires an artifact service to be configured")
	}
	if err := toolutils.PackTool(req, t); err != nil {
		return err
	}
	listed, err := ctx.Artifacts().List(ctx)
	if err != nil {
		return fmt.Errorf("failed to list artifacts: %w", err)
	}
	if err := appendInitialInstructions(req, listed.FileNames); err != nil {
		return err
	}
	return t.processLoadArtifactsFunctionCall(ctx, req, listed.FileNames)
}

func appendInitialInstructions(req *model.LLMRequest, artifactNames []string) error {
	if len(artifactNames) == 0 {
		return nil
	}
	artifactNamesJSON, err := json.Marshal(artifactNames)
	if err != nil {
		return fmt.Errorf("failed to marshal artifact names: %w", err)
	}
	instructions := fmt.Sprintf(
		"You have a list of artifacts:\n  %s\n\nWhen the user asks questions about"+
			" any of the artifacts, you should call the `load_artifacts` function"+
			" to load the artifact. Do not generate any text other than the"+
			" function call. Whenever you are asked about artifacts, you"+
			" should first load it. You must always load an artifact to access its"+
			" content, even if it has been loaded before.", string(artifactNamesJSON))

	utils.AppendInstructions(req, instructions)
	return nil
}

func (t *artifactsTool) processLoadArtifactsFunctionCall(ctx agent.Context, req *model.LLMRequest, availableNames []string) error {
	if len(req.Contents) == 0 {
		return nil
	}
	responses := slices.DeleteFunc(utils.FunctionResponses(req.Contents[len(req.Contents)-1]), func(response *genai.FunctionResponse) bool {
		return response.Name != t.Name()
	})
	// Like Python, keep other calls' results when a parallel response is
	// malformed or names an unavailable artifact. Preserve Go's single-call errors.
	parallel := len(responses) > 1
	var artifactNames []string
	for _, response := range responses {
		raw, ok := response.Response["artifact_names"]
		if !ok {
			// Invalid arguments produce a tool error result without artifact_names.
			continue
		}
		names, err := artifactNamesFromResponse(raw)
		if err != nil {
			if parallel {
				continue
			}
			return err
		}
		artifactNames = append(artifactNames, names...)
	}
	if parallel {
		artifactNames = slices.DeleteFunc(artifactNames, func(name string) bool {
			return !slices.Contains(availableNames, name)
		})
	}
	if len(artifactNames) == 0 {
		return nil
	}

	results := make([]*genai.Content, len(artifactNames))
	group, childCtx := errgroup.WithContext(ctx)
	artifactsService := ctx.Artifacts()

	for i, artifactName := range artifactNames {
		group.Go(func() error {
			content, err := t.loadIndividualArtifact(childCtx, artifactsService, artifactName)
			if err != nil {
				// The file may have been removed after List completed.
				if parallel && errors.Is(err, fs.ErrNotExist) {
					return nil
				}
				return err
			}
			results[i] = content
			return nil
		})
	}

	if err := group.Wait(); err != nil {
		return err
	}

	for _, content := range results {
		if content != nil {
			req.Contents = append(req.Contents, content)
		}
	}
	return nil
}

func artifactNamesFromResponse(raw any) ([]string, error) {
	switch names := raw.(type) {
	case []string:
		return names, nil
	case []any:
		result := make([]string, len(names))
		for i, name := range names {
			s, ok := name.(string)
			if !ok {
				return nil, fmt.Errorf("invalid artifact name type at index %d: %T, expected string", i, name)
			}
			result[i] = s
		}
		return result, nil
	default:
		return nil, fmt.Errorf("invalid artifact names type: %T, expected []string or []any", raw)
	}
}

func (t *artifactsTool) loadIndividualArtifact(ctx context.Context, artifactsService agent.Artifacts, artifactName string) (*genai.Content, error) {
	resp, err := artifactsService.Load(ctx, artifactName)
	if err != nil {
		return nil, fmt.Errorf("failed to load artifact: %w", err)
	}
	return &genai.Content{
		Parts: []*genai.Part{
			genai.NewPartFromText("Artifact " + artifactName + " is:"),
			resp.Part,
		},
		Role: genai.RoleUser,
	}, nil
}
