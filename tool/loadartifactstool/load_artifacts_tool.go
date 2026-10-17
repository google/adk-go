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
	"strings"

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
		if err := json.Unmarshal(artifactNamesJson, &artifactNames); err != nil {
			return nil, fmt.Errorf("failed to unmarshal artifact_names from JSON to []string: %w", err)
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

// ProcessRequest processes the LLM request. It packs the tool, appends initial
// instructions, and processes any load artifacts function calls.
func (t *artifactsTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	if ctx.Artifacts() == nil {
		return fmt.Errorf("load_artifacts tool requires an artifact service to be configured")
	}
	if err := toolutils.PackTool(req, t); err != nil {
		return err
	}
	if err := t.appendInitialInstructions(ctx, req); err != nil {
		return err
	}
	return t.processLoadArtifactsFunctionCall(ctx, req)
}

func (t *artifactsTool) appendInitialInstructions(ctx agent.Context, req *model.LLMRequest) error {
	resp, err := ctx.Artifacts().List(ctx)
	if err != nil {
		return fmt.Errorf("failed to list artifacts: %w", err)
	}
	if len(resp.FileNames) == 0 {
		return nil
	}
	artifactNamesJSON, err := json.Marshal(resp.FileNames)
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

func (t *artifactsTool) processLoadArtifactsFunctionCall(ctx agent.Context, req *model.LLMRequest) error {
	if len(req.Contents) == 0 {
		return nil
	}
	lastContent := req.Contents[len(req.Contents)-1]
	if lastContent == nil || len(lastContent.Parts) == 0 {
		return nil
	}
	var functionResponse *genai.FunctionResponse
	// Iterate over all parts in the last content turn to find load_artifacts responses.
	// Note: adk-python only checks parts[0]; scanning all parts is intentional in Go to
	// support parallel/multi-tool turns where load_artifacts may not be the first part.
	for _, part := range lastContent.Parts {
		if part != nil && part.FunctionResponse != nil && part.FunctionResponse.Name == "load_artifacts" {
			functionResponse = part.FunctionResponse
			// Keep only the first load_artifacts response if multiple exist in one turn.
			break
		}
	}
	if functionResponse == nil {
		return nil
	}
	artifactNamesRaw, ok := functionResponse.Response["artifact_names"]
	if !ok {
		return nil
	}
	var artifactNames []string
	switch names := artifactNamesRaw.(type) {
	case []string:
		artifactNames = names
	case []any:
		artifactNames = make([]string, len(names))
		for i, name := range names {
			s, ok := name.(string)
			if !ok {
				return fmt.Errorf("invalid artifact name type at index %d: %T, expected string", i, name)
			}
			artifactNames[i] = s
		}
	default:
		return fmt.Errorf("invalid artifact names type: %T, expected []string or []any", artifactNamesRaw)
	}
	if len(artifactNames) == 0 {
		return nil
	}

	results := make([]*genai.Content, len(artifactNames))
	// notes[i] explains why artifactNames[i] was not loaded. Per index rather
	// than an appended list so the goroutines never share a variable.
	notes := make([]string, len(artifactNames))
	group, childCtx := errgroup.WithContext(ctx)
	artifactsService := ctx.Artifacts()

	for i, artifactName := range artifactNames {
		group.Go(func() error {
			// The load runs here, on the request that follows the tool call, so
			// returning a recoverable failure from ProcessRequest ends the turn
			// before the model is asked again, and the model never learns what
			// went wrong. A name the model got wrong is something it can fix, so
			// those are reported back to it as content. Anything it cannot fix --
			// a storage failure, a cancelled context -- is still returned.
			if !isValidArtifactName(artifactName) {
				notes[i] = fmt.Sprintf("%q is not a valid artifact name: artifact names are plain file names and cannot contain path separators", artifactName)
				return nil
			}
			// Although not used, we need to pass childCtx for early return in case of an error.
			content, err := t.loadIndividualArtifact(childCtx, artifactsService, artifactName)
			if err != nil {
				if errors.Is(err, fs.ErrNotExist) {
					notes[i] = fmt.Sprintf("%q was not found", artifactName)
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
	var failed []string
	for _, note := range notes {
		if note != "" {
			failed = append(failed, note)
		}
	}
	if len(failed) > 0 {
		// Tell the model which artifacts it did not get and why, so it can
		// correct the call or answer without them instead of the turn ending.
		req.Contents = append(req.Contents, &genai.Content{
			Role:  genai.RoleUser,
			Parts: []*genai.Part{genai.NewPartFromText(strings.Join(failed, "\n"))},
		})
	}
	return nil
}

// isValidArtifactName reports whether name can be an artifact file name. The
// artifact service rejects names containing path separators; checking that here
// lets the failure be reported to the model instead of ending the turn.
func isValidArtifactName(name string) bool {
	return !strings.Contains(name, "/") && !strings.Contains(name, "\\")
}

func (t *artifactsTool) loadIndividualArtifact(ctx context.Context, artifactsService agent.Artifacts, artifactName string) (*genai.Content, error) {
	resp, err := artifactsService.Load(ctx, artifactName)
	if err != nil {
		return nil, fmt.Errorf("failed to load artifact %s: %w", artifactName, err)
	}
	return &genai.Content{
		Parts: []*genai.Part{
			genai.NewPartFromText("Artifact " + artifactName + " is:"),
			resp.Part,
		},
		Role: genai.RoleUser,
	}, nil
}
