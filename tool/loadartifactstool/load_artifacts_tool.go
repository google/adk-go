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
	"fmt"
	"math"
	"slices"
	"time"

	"golang.org/x/sync/errgroup"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
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
					Type:        "ARRAY",
					Description: "Names of artifacts to load, or whose revisions to list when list_versions is true.",
					Items: &genai.Schema{
						Type: "STRING",
					},
				},
				"version": {
					Type:        "INTEGER",
					Description: "Optional 1-based revision number to load for each artifact in artifact_names. If omitted or 0, the latest revision is loaded.",
				},
				"artifact_versions": {
					Type:        "ARRAY",
					Description: "Optional list of specific artifact revisions to load (for example, to load or compare two revisions in one call).",
					Items: &genai.Schema{
						Type: "OBJECT",
						Properties: map[string]*genai.Schema{
							"name": {
								Type:        "STRING",
								Description: "Artifact filename.",
							},
							"version": {
								Type:        "INTEGER",
								Description: "1-based revision number to load. If omitted or 0, the latest revision is loaded.",
							},
						},
						Required: []string{"name"},
					},
				},
				"list_versions": {
					Type:        "BOOLEAN",
					Description: "If true, lists the available revisions and their metadata for each artifact in artifact_names instead of loading content.",
				},
			},
		},
	}
}

type artifactVersionSelector struct {
	Name    string `json:"name"`
	Version int    `json:"version,omitempty"`
}

// Run implements tool.Tool.
func (t *artifactsTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type, got: %T", args)
	}
	artifactNames, err := parseStringSliceArg(m, "artifact_names")
	if err != nil {
		return nil, err
	}
	result := map[string]any{
		"artifact_names": artifactNames,
	}

	if rawVersion, exists := m["version"]; exists && rawVersion != nil {
		version, err := parseNonNegativeInt(rawVersion, "version")
		if err != nil {
			return nil, err
		}
		if version > 0 {
			result["version"] = version
		}
	}

	if rawSelectors, exists := m["artifact_versions"]; exists && rawSelectors != nil {
		selectors, err := parseArtifactVersionSelectors(rawSelectors)
		if err != nil {
			return nil, err
		}
		if len(selectors) > 0 {
			encoded := make([]map[string]any, len(selectors))
			for i, sel := range selectors {
				encoded[i] = map[string]any{
					"name":    sel.Name,
					"version": sel.Version,
				}
			}
			result["artifact_versions"] = encoded
		}
	}

	if rawListVersions, exists := m["list_versions"]; exists && rawListVersions != nil {
		listVersions, extraNames, err := parseListVersionsArg(rawListVersions)
		if err != nil {
			return nil, err
		}
		for _, name := range extraNames {
			if !slices.Contains(artifactNames, name) {
				artifactNames = append(artifactNames, name)
			}
		}
		result["artifact_names"] = artifactNames
		if listVersions {
			result["list_versions"] = true
		}
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

	var artifactNames []string
	if artifactNamesRaw, ok := functionResponse.Response["artifact_names"]; ok && artifactNamesRaw != nil {
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
	}

	version := 0
	if rawVersion, ok := functionResponse.Response["version"]; ok && rawVersion != nil {
		var err error
		version, err = parseNonNegativeInt(rawVersion, "version")
		if err != nil {
			return err
		}
	}

	var selectors []artifactVersionSelector
	if rawSelectors, ok := functionResponse.Response["artifact_versions"]; ok && rawSelectors != nil {
		var err error
		selectors, err = parseArtifactVersionSelectors(rawSelectors)
		if err != nil {
			return err
		}
	}

	listVersions := false
	if rawListVersions, ok := functionResponse.Response["list_versions"]; ok && rawListVersions != nil {
		var extraNames []string
		var err error
		listVersions, extraNames, err = parseListVersionsArg(rawListVersions)
		if err != nil {
			return err
		}
		for _, name := range extraNames {
			if !slices.Contains(artifactNames, name) {
				artifactNames = append(artifactNames, name)
			}
		}
	}

	artifactsService := ctx.Artifacts()
	if listVersions {
		if len(artifactNames) == 0 {
			return nil
		}
		results := make([]*genai.Content, len(artifactNames))
		group, childCtx := errgroup.WithContext(ctx)
		for i, artifactName := range artifactNames {
			group.Go(func() error {
				content, err := t.listArtifactVersions(childCtx, artifactsService, artifactName)
				if err != nil {
					return err
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

	loadItems := make([]artifactVersionSelector, 0, len(artifactNames)+len(selectors))
	for _, name := range artifactNames {
		loadItems = append(loadItems, artifactVersionSelector{Name: name, Version: version})
	}
	loadItems = append(loadItems, selectors...)
	if len(loadItems) == 0 {
		return nil
	}

	results := make([]*genai.Content, len(loadItems))
	group, childCtx := errgroup.WithContext(ctx)

	for i, item := range loadItems {
		group.Go(func() error {
			content, err := t.loadIndividualArtifact(childCtx, artifactsService, item.Name, item.Version)
			if err != nil {
				return err
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

func (t *artifactsTool) loadIndividualArtifact(ctx context.Context, artifactsService agent.Artifacts, artifactName string, version int) (*genai.Content, error) {
	if version > 0 {
		resp, err := artifactsService.LoadVersion(ctx, artifactName, version)
		if err != nil {
			return nil, fmt.Errorf("failed to load artifact %s (version %d): %w", artifactName, version, err)
		}
		return &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText(fmt.Sprintf("Artifact %s (version %d) is:", artifactName, version)),
				resp.Part,
			},
			Role: genai.RoleUser,
		}, nil
	}
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

type artifactVersionSummary struct {
	Version    int64  `json:"version"`
	CreateTime string `json:"create_time,omitempty"`
	MimeType   string `json:"mime_type,omitempty"`
}

func (t *artifactsTool) listArtifactVersions(ctx context.Context, artifactsService agent.Artifacts, artifactName string) (*genai.Content, error) {
	versionsResp, err := artifactinternal.Versions(ctx, artifactsService, artifactName)
	if err != nil {
		return nil, fmt.Errorf("failed to list versions for artifact %s: %w", artifactName, err)
	}
	versions := slices.Clone(versionsResp.Versions)
	slices.Sort(versions)

	summaries := make([]artifactVersionSummary, 0, len(versions))
	for _, v := range versions {
		metaResp, err := artifactinternal.GetArtifactVersion(ctx, artifactsService, artifactName, int(v))
		if err != nil {
			return nil, fmt.Errorf("failed to get version %d metadata for artifact %s: %w", v, artifactName, err)
		}
		summary := artifactVersionSummary{Version: v}
		if av := metaResp.ArtifactVersion; av != nil {
			summary.Version = av.Version
			if !av.CreateTime.IsZero() {
				summary.CreateTime = av.CreateTime.UTC().Format(time.RFC3339Nano)
			}
			summary.MimeType = av.MimeType
		}
		summaries = append(summaries, summary)
	}

	encoded, err := json.Marshal(summaries)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal version metadata for artifact %s: %w", artifactName, err)
	}
	return &genai.Content{
		Parts: []*genai.Part{
			genai.NewPartFromText(fmt.Sprintf("Versions of artifact %s:", artifactName)),
			genai.NewPartFromText(string(encoded)),
		},
		Role: genai.RoleUser,
	}, nil
}

func parseStringSliceArg(m map[string]any, key string) ([]string, error) {
	raw, exists := m[key]
	if !exists || raw == nil {
		return []string{}, nil
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal %s to JSON: %w", key, err)
	}
	var out []string
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("failed to unmarshal %s from JSON to []string: %w", key, err)
	}
	if out == nil {
		out = []string{}
	}
	return out, nil
}

func parseNonNegativeInt(raw any, field string) (int, error) {
	var v int
	switch n := raw.(type) {
	case int:
		v = n
	case int64:
		v = int(n)
	case float64:
		if math.IsNaN(n) || math.IsInf(n, 0) || math.Trunc(n) != n || n < 0 || n > float64(math.MaxInt32) {
			return 0, fmt.Errorf("invalid %s: must be a non-negative integer", field)
		}
		v = int(n)
	case json.Number:
		i64, err := n.Int64()
		if err != nil {
			return 0, fmt.Errorf("invalid %s: %w", field, err)
		}
		v = int(i64)
	default:
		return 0, fmt.Errorf("invalid %s type: %T, expected integer", field, raw)
	}
	if v < 0 {
		return 0, fmt.Errorf("invalid %s: %d must be non-negative", field, v)
	}
	return v, nil
}

func parseArtifactVersionSelectors(raw any) ([]artifactVersionSelector, error) {
	var items []any
	switch v := raw.(type) {
	case []any:
		items = v
	case []map[string]any:
		items = make([]any, len(v))
		for i, item := range v {
			items[i] = item
		}
	default:
		return nil, fmt.Errorf("invalid artifact_versions type: %T, expected array of objects", raw)
	}

	out := make([]artifactVersionSelector, len(items))
	for i, item := range items {
		obj, ok := item.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("invalid artifact_versions[%d] type: %T, expected object", i, item)
		}
		nameRaw, ok := obj["name"]
		if !ok {
			return nil, fmt.Errorf("missing required field \"name\" in artifact_versions[%d]", i)
		}
		name, ok := nameRaw.(string)
		if !ok || name == "" {
			return nil, fmt.Errorf("invalid \"name\" in artifact_versions[%d]: expected non-empty string", i)
		}
		version := 0
		if verRaw, exists := obj["version"]; exists && verRaw != nil {
			var err error
			version, err = parseNonNegativeInt(verRaw, fmt.Sprintf("artifact_versions[%d].version", i))
			if err != nil {
				return nil, err
			}
		}
		out[i] = artifactVersionSelector{Name: name, Version: version}
	}
	return out, nil
}

func parseListVersionsArg(raw any) (bool, []string, error) {
	if b, ok := raw.(bool); ok {
		return b, nil, nil
	}
	names, err := parseStringSliceArg(map[string]any{"list_versions": raw}, "list_versions")
	if err != nil {
		return false, nil, fmt.Errorf("invalid list_versions: expected bool or []string: %w", err)
	}
	return len(names) > 0, names, nil
}
