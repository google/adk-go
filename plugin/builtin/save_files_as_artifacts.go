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

package builtin

import (
	"fmt"
	"log/slog"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/plugin"
)

// SaveFilesAsArtifactsConfig configures NewSaveFilesAsArtifacts.
type SaveFilesAsArtifactsConfig struct {
	// Name overrides the default plugin name.
	Name string

	// DontAttachFileReference, when true, saves files silently without
	// replacing inline-data parts in the user message. Default zero
	// value (false) preserves the upstream behavior: the plugin
	// substitutes a textual placeholder ("Uploaded file: <name>. Saved
	// as artifact.") so the model knows where to find each file.
	// Inverted from attach_file_reference in adk-python so the Go zero
	// value matches Python's default of True.
	DontAttachFileReference bool

	// Logger is used for warnings (e.g. when artifact service isn't
	// configured). Defaults to slog.Default().
	Logger *slog.Logger
}

// NewSaveFilesAsArtifacts builds a plugin that intercepts user
// messages, saves every inline-data part as an artifact, and (by
// default) replaces each saved part with a text placeholder so the
// model knows where to find the file. Mirrors adk-python's
// SaveFilesAsArtifactsPlugin.
//
// When the artifact service is not configured on the runner, the
// plugin emits a warning and passes the message through unchanged.
//
// File names: the part's DisplayName is used when present; otherwise
// the plugin generates "artifact_<invocation_id>_<index>".
func NewSaveFilesAsArtifacts(cfg SaveFilesAsArtifactsConfig) (*plugin.Plugin, error) {
	name := cfg.Name
	if name == "" {
		name = "save_files_as_artifacts"
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	// Match adk-python default: attach is on unless caller opts out.
	attach := !cfg.DontAttachFileReference

	return plugin.New(plugin.Config{
		Name: name,
		OnUserMessageCallback: func(ic agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
			if msg == nil || len(msg.Parts) == 0 {
				return nil, nil
			}
			art := ic.Artifacts()
			if art == nil {
				logger.Warn("save_files_as_artifacts: artifact service not configured; pass-through",
					slog.String("plugin", name))
				return nil, nil
			}
			modified := false
			newParts := make([]*genai.Part, 0, len(msg.Parts))
			for i, p := range msg.Parts {
				if p == nil || p.InlineData == nil {
					newParts = append(newParts, p)
					continue
				}
				fileName := p.InlineData.DisplayName
				if fileName == "" {
					fileName = fmt.Sprintf("artifact_%s_%d", ic.InvocationID(), i)
				}
				saved := &genai.Part{InlineData: p.InlineData}
				if _, err := art.Save(ic, fileName, saved); err != nil {
					logger.Error("save_files_as_artifacts: save failed",
						slog.String("plugin", name),
						slog.String("file", fileName),
						slog.String("err", err.Error()),
					)
					newParts = append(newParts, p)
					continue
				}
				modified = true
				if attach {
					newParts = append(newParts, &genai.Part{
						Text: fmt.Sprintf("Uploaded file: %s. It has been saved to the artifacts.", fileName),
					})
				}
			}
			if !modified {
				return nil, nil
			}
			return &genai.Content{Role: msg.Role, Parts: newParts}, nil
		},
	})
}
