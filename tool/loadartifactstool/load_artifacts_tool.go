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
	"mime"
	"strings"

	"golang.org/x/sync/errgroup"
	"golang.org/x/text/encoding/ianaindex"

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

func (t *artifactsTool) loadIndividualArtifact(ctx context.Context, artifactsService agent.Artifacts, artifactName string) (*genai.Content, error) {
	resp, err := artifactsService.Load(ctx, artifactName)
	if err != nil {
		return nil, fmt.Errorf("failed to load artifact %s: %w", artifactName, err)
	}
	return &genai.Content{
		Parts: []*genai.Part{
			genai.NewPartFromText("Artifact " + artifactName + " is:"),
			safePartForLLM(resp.Part, artifactName),
		},
		Role: genai.RoleUser,
	}, nil
}

// normalizeMIMEType drops any parameters and control characters and lowercases
// the type, which is case-insensitive per RFC 2045. Control characters never
// appear in a legitimate label but would otherwise dodge the exact-match lists
// below while still hitting the prefix matches.
func normalizeMIMEType(mimeType string) string {
	mimeType, _, _ = strings.Cut(mimeType, ";")
	mimeType = strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f {
			return -1
		}
		return r
	}, mimeType)
	return strings.ToLower(strings.TrimSpace(mimeType))
}

// isInlineMIMETypeSupported reports whether the model accepts this type inline.
// The argument must already be normalized; the call site does so. The media
// match is prefix-based on purpose: enumerating subtypes would rot.
func isInlineMIMETypeSupported(mimeType string) bool {
	// Gemini rejects SVG source as inline image data whatever subtype it is
	// labelled with (the rejection is on the bytes, not the label), so this
	// list catches the conventional labels for such artifacts rather than
	// mirroring a backend rule. The set matches adk-python's
	// _GEMINI_UNSUPPORTED_INLINE_SUBTYPES.
	switch mimeType {
	case "image/svg", "image/svg+xml", "image/xml":
		return false
	}
	return strings.HasPrefix(mimeType, "image/") ||
		strings.HasPrefix(mimeType, "audio/") ||
		strings.HasPrefix(mimeType, "video/") ||
		mimeType == "application/pdf"
}

// isTextLikeMIMEType reports whether data of this type can be inlined as text.
// The argument must already be normalized; the call site does so.
func isTextLikeMIMEType(mimeType string) bool {
	switch mimeType {
	case "application/csv",
		"application/json",
		"application/svg+xml",
		"application/xml",
		"image/svg",
		"image/svg+xml",
		"image/xml":
		return true
	}
	return strings.HasPrefix(mimeType, "text/")
}

// decodeArtifactText converts artifact bytes to UTF-8 text. A charset
// parameter on the original MIME type is honoured when it names a known
// encoding, so "text/csv; charset=windows-1252" round-trips instead of
// degrading; otherwise invalid sequences are replaced rather than the
// artifact discarded.
func decodeArtifactText(data []byte, rawMIMEType string) string {
	if _, params, err := mime.ParseMediaType(rawMIMEType); err == nil {
		if cs := params["charset"]; cs != "" && !strings.EqualFold(cs, "utf-8") {
			if enc, err := ianaindex.IANA.Encoding(cs); err == nil && enc != nil {
				if decoded, err := enc.NewDecoder().Bytes(data); err == nil {
					return string(decoded)
				}
			}
		}
	}
	return strings.ToValidUTF8(string(data), "\uFFFD")
}

// safePartForLLM returns a part the model will accept, converting or describing
// inline data it would reject. The result is never nil.
func safePartForLLM(part *genai.Part, artifactName string) *genai.Part {
	if part == nil {
		return genai.NewPartFromText(fmt.Sprintf("[Artifact: %q. No content was returned.]", artifactName))
	}
	if part.InlineData == nil {
		return part
	}

	mimeType := normalizeMIMEType(part.InlineData.MIMEType)
	if mimeType == "" {
		mimeType = "application/octet-stream"
	}
	data := part.InlineData.Data
	// An empty blob is degenerate whatever its declared type.
	if len(data) == 0 {
		return genai.NewPartFromText(fmt.Sprintf("[Artifact: %q, type: %q. No inline data was provided.]", artifactName, mimeType))
	}
	if isInlineMIMETypeSupported(mimeType) {
		return part
	}
	// Text in a legacy encoding still carries readable content, so decode it
	// rather than discarding the artifact.
	if isTextLikeMIMEType(mimeType) {
		return genai.NewPartFromText(decodeArtifactText(data, part.InlineData.MIMEType))
	}

	return genai.NewPartFromText(fmt.Sprintf(
		"[Binary artifact: %q, type: %q, size: %d bytes. Content cannot be displayed inline.]",
		artifactName,
		mimeType,
		len(data),
	))
}
