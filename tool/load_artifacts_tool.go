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
	"context"
	"encoding/json"
	"fmt"

	"golang.org/x/sync/errgroup"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/toolinternal/toolutils"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// loadArtifactsTool is a tool that loads artifacts and adds them to the session.
type loadArtifactsTool struct {
	name        string
	description string
}

// NewLoadArtifactsTool creates a new loadArtifactsTool.
func NewLoadArtifactsTool() Tool {
	return &loadArtifactsTool{
		name:        "load_artifacts",
		description: "Loads the artifacts and adds them to the session.",
	}
}

// Name implements tool.Tool.
func (t *loadArtifactsTool) Name() string {
	return t.name
}

// Description implements tool.Tool.
func (t *loadArtifactsTool) Description() string {
	return t.description
}

// IsLongRunning implements tool.Tool.
func (t *loadArtifactsTool) IsLongRunning() bool {
	return false
}

// Declaration implements tool.Tool.
func (t *loadArtifactsTool) Declaration() *genai.FunctionDeclaration {
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
func (t *loadArtifactsTool) Run(ctx Context, args any) (any, error) {
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

// ProcessRequest implements tool.Tool.
func (t *loadArtifactsTool) ProcessRequest(ctx Context, req *model.LLMRequest) error {
	if err := toolutils.PackTool(req, t); err != nil {
		return err
	}
	if err := t.appendInitialInstructions(ctx, req); err != nil {
		return err
	}
	return t.processLoadArtifactsFunctionCall(ctx, req)
}

func (t *loadArtifactsTool) appendInitialInstructions(ctx Context, req *model.LLMRequest) error {
	artifactNames, err := ctx.Artifacts().List()
	if err != nil {
		return fmt.Errorf("failed to list artifacts: %w", err)
	}
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

func (t *loadArtifactsTool) processLoadArtifactsFunctionCall(ctx Context, req *model.LLMRequest) error {
	if len(req.Contents) == 0 {
		return nil
	}
	lastContent := req.Contents[len(req.Contents)-1]
	if lastContent == nil || len(lastContent.Parts) == 0 {
		return nil
	}
	firstPart := lastContent.Parts[0]
	if firstPart.FunctionResponse == nil {
		return nil
	}

	functionResponse := firstPart.FunctionResponse

	if functionResponse.Name != "load_artifacts" {
		return nil
	}
	artifactNamesRaw, ok := functionResponse.Response["artifact_names"]
	if !ok {
		return nil
	}
	artifactNames, ok := artifactNamesRaw.([]string)
	if !ok {
		return fmt.Errorf("invalid artifact names type: %T, expected []string", artifactNamesRaw)
	}
	if len(artifactNames) == 0 {
		return nil
	}

	results := make([]*genai.Content, len(artifactNames))
	group, childCtx := errgroup.WithContext(ctx)
	artifactsService := ctx.Artifacts()

	for i, artifactName := range artifactNames {
		group.Go(func() error {
			// Although not used, we need to pass childCtx for early return in case of an error.
			content, err := t.loadIndividualArtifact(childCtx, artifactsService, artifactName)
			if err != nil {
				return fmt.Errorf("failed to load artifact %s: %w", artifactName, err)
			}
			results[i] = content
			return nil
		})
	}

	if err := group.Wait(); err != nil {
		return err
	}

	req.Contents = append(req.Contents, results...)
	return nil
}

func (t *loadArtifactsTool) loadIndividualArtifact(_ context.Context, artifactsService agent.Artifacts, artifactName string) (*genai.Content, error) {
	artifact, err := artifactsService.Load(artifactName)
	if err != nil {
		return nil, fmt.Errorf("failed to load artifact %s: %w", artifactName, err)
	}
	return &genai.Content{
		Parts: []*genai.Part{
			genai.NewPartFromText("Artifact " + artifactName + " is:"),
			&artifact,
		},
		Role: genai.RoleUser,
	}, nil
}
