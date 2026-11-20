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

// Package savefilesasartifacts provides a plugin that saves files embedded in
// user messages as artifacts.
//
// Each part of the user message that carries [genai.Blob] inline data is saved
// as an artifact named after the blob's DisplayName, or
// artifact_<invocationID>_<partIndex> when the blob has none, and is replaced
// in the message by a text placeholder naming the artifact. By default
// artifacts are session-scoped; prefix the display name with "user:" to keep
// the file across sessions. A file with the same name overwrites the previous
// one as a new version. Give the agent a tool that loads artifacts, such as
// loadartifactstool, so the model can read the files.
//
// The saved versions are added to the ArtifactDelta of the first non-partial
// event of the invocation, so they appear in the session's artifact history.
// Until then they are held in the run's session state under a temp: key, which
// is never persisted, so nothing outlives the run. When the invocation commits
// no event, they are not reported.
//
// The runner stops calling OnUserMessageCallback, and OnEventCallback, at the
// first plugin that returns a non-nil replacement, and skips committing an
// event when a plugin returns an error. Plugin order therefore costs:
//   - Registered after a plugin that rewrites the user message, this plugin
//     never sees the message, and its files are not saved.
//   - Registered after a plugin that replaces an event, this plugin never sees
//     that event, and the versions are reported on the next one it does see.
//   - Registered before a plugin that returns an error for an event, the
//     versions are written into an event that is not committed, and are not
//     reported.
//
// Do not combine this plugin with [agent.RunConfig.SaveInputBlobsAsArtifacts].
// When a save fails here, the original part is kept in the message, and the
// runner then saves it again and fails the run if that save also fails.
package savefilesasartifacts

import (
	"fmt"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

const defaultName = "save_files_as_artifacts_plugin"

// maxInlineDataSize is the largest inline file the plugin saves, matching the
// Gemini API limit for inline data (https://ai.google.dev/gemini-api/docs/files).
const maxInlineDataSize = 20 * 1024 * 1024

// modelAccessibleURISchemes are the artifact URI schemes current model
// connectors can read: gs:// on Vertex AI, http(s):// on hosted endpoints.
var modelAccessibleURISchemes = map[string]bool{"gs": true, "https": true, "http": true}

// Config configures the plugin created by [New].
type Config struct {
	// Name is the plugin name. The zero value means
	// "save_files_as_artifacts_plugin".
	Name string

	// SkipFileReference disables the FileData part that otherwise follows the
	// placeholder of each saved file whose canonical URI the model can read
	// (gs, http or https). The file is still saved as an artifact.
	SkipFileReference bool

	// ArtifactService is used only to look up the canonical URI of a saved
	// version for the FileData reference, and should be the same service the
	// runner saves artifacts to. When nil, no FileData reference is added.
	ArtifactService artifact.Service
}

// New creates a plugin that saves the inline files of each user message as
// artifacts, as described in the package documentation.
//
// When the invocation has no artifact service, the user message is left
// unchanged.
func New(cfg Config) (*plugin.Plugin, error) {
	if cfg.Name == "" {
		cfg.Name = defaultName
	}
	p := &saveFilesPlugin{
		cfg:        cfg,
		pendingKey: session.KeyPrefixTemp + cfg.Name + ":pending_delta",
	}
	return plugin.New(plugin.Config{
		Name:                  cfg.Name,
		OnUserMessageCallback: p.onUserMessage,
		OnEventCallback:       p.onEvent,
	})
}

// saveFilesPlugin holds no per-run state: the versions saved for a run are
// kept in that run's session state, as adk-python does.
type saveFilesPlugin struct {
	cfg Config
	// pendingKey is the session state key that holds the versions saved by
	// onUserMessage until onEvent records them in an event's actions.
	pendingKey string
}

func (p *saveFilesPlugin) onUserMessage(ctx agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
	artifacts := ctx.Artifacts()
	if artifacts == nil || msg == nil {
		return nil, nil
	}

	var newParts []*genai.Part
	saved := make(map[string]int64)
	modified := false

	for i, part := range msg.Parts {
		if part == nil || part.InlineData == nil {
			newParts = append(newParts, part)
			continue
		}
		blob := part.InlineData

		name := blob.DisplayName
		if name == "" {
			name = fmt.Sprintf("artifact_%s_%d", ctx.InvocationID(), i)
		}

		if size := len(blob.Data); size > maxInlineDataSize {
			errMsg := fmt.Sprintf(
				"File %s (%.2f MB) exceeds the maximum supported size of %.0fMB. Please upload a smaller file.",
				name, float64(size)/(1024*1024), float64(maxInlineDataSize)/(1024*1024),
			)
			newParts = append(newParts, genai.NewPartFromText("[Upload Error: "+errMsg+"]"))
			modified = true
			continue
		}

		// Save a shallow copy so later changes to the message part's fields do
		// not alter the stored artifact. The blob's Data is still shared with
		// the caller, as in adk-python.
		partCopy := *part
		resp, err := artifacts.Save(ctx, name, &partCopy)
		if err != nil {
			// Keep the original part so one failed save does not lose the file
			// or fail the run.
			newParts = append(newParts, part)
			continue
		}

		newParts = append(newParts, genai.NewPartFromText(`[Uploaded Artifact: "`+name+`"]`))
		if fileRef := p.fileReference(ctx, name, resp.Version, blob.MIMEType); fileRef != nil {
			newParts = append(newParts, fileRef)
		}
		saved[name] = resp.Version
		modified = true
	}

	if !modified {
		return nil, nil
	}

	if len(saved) > 0 {
		if sess := ctx.Session(); sess != nil {
			if err := sess.State().Set(p.pendingKey, saved); err != nil {
				return nil, fmt.Errorf("record saved artifact versions: %w", err)
			}
		}
	}

	return &genai.Content{Role: msg.Role, Parts: newParts}, nil
}

// onEvent adds the versions saved for this invocation to the ArtifactDelta of
// its first non-partial event, the first one the runner commits to the
// session. The event is changed in place so later plugins still see it.
func (p *saveFilesPlugin) onEvent(ctx agent.InvocationContext, ev *session.Event) (*session.Event, error) {
	if ev == nil || ev.Partial {
		return nil, nil
	}
	sess := ctx.Session()
	if sess == nil {
		return nil, nil
	}
	v, err := sess.State().Get(p.pendingKey)
	if err != nil {
		return nil, nil
	}
	delta, _ := v.(map[string]int64)
	if len(delta) == 0 {
		return nil, nil
	}
	if err := sess.State().Set(p.pendingKey, map[string]int64(nil)); err != nil {
		return nil, fmt.Errorf("clear saved artifact versions: %w", err)
	}
	if ev.Actions.ArtifactDelta == nil {
		ev.Actions.ArtifactDelta = make(map[string]int64)
	}
	for name, version := range delta {
		// Keep a newer version the agent saved under the same name.
		if v, ok := ev.Actions.ArtifactDelta[name]; !ok || version > v {
			ev.Actions.ArtifactDelta[name] = version
		}
	}
	return nil, nil
}

// fileReference returns a FileData part pointing at the saved version, or nil
// when references are disabled, no artifact service is configured, or the
// version's canonical URI is not one the model can read.
func (p *saveFilesPlugin) fileReference(ctx agent.InvocationContext, name string, version int64, mimeType string) *genai.Part {
	if p.cfg.SkipFileReference || p.cfg.ArtifactService == nil {
		return nil
	}
	sess := ctx.Session()
	if sess == nil {
		return nil
	}
	resp, err := p.cfg.ArtifactService.GetArtifactVersion(ctx, &artifact.GetArtifactVersionRequest{
		AppName:   sess.AppName(),
		UserID:    sess.UserID(),
		SessionID: sess.ID(),
		FileName:  name,
		Version:   version,
	})
	if err != nil || resp == nil || resp.ArtifactVersion == nil {
		return nil
	}
	av := resp.ArtifactVersion
	if !isModelAccessibleURI(av.CanonicalURI) {
		return nil
	}
	if mimeType == "" {
		mimeType = av.MimeType
	}
	return &genai.Part{FileData: &genai.FileData{
		FileURI:     av.CanonicalURI,
		MIMEType:    mimeType,
		DisplayName: name,
	}}
}

// isModelAccessibleURI reports whether uri has a scheme the model can read.
// Only the scheme is checked: the GCS service writes object names into the
// URI unescaped, so a full url.Parse would reject a name such as "50% off.pdf".
func isModelAccessibleURI(uri string) bool {
	scheme, _, ok := strings.Cut(uri, ":")
	return ok && modelAccessibleURISchemes[strings.ToLower(scheme)]
}
