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

// Package savefilesasartifacts provides a plugin that saves the files a user
// attaches to a message as artifacts of the session, so that a chat user can
// upload a file once and the agent's tools can load it for the rest of the
// session.
//
// Register the plugin that [New] returns with
// [google.golang.org/adk/v2/runner.Config.PluginConfig], on a runner that has
// an artifact service. Each inline file of a user message is saved under its
// display name and replaced in the message by a text placeholder such as
// [Uploaded Artifact: "report.pdf"], optionally followed by a file reference
// the model can read. The versions saved are reported in the artifact delta of
// the first agent event of the invocation. Give the agent a tool that loads
// artifacts, or load them in your own tools, to use the files.
//
// The plugin is the counterpart of adk-python's SaveFilesAsArtifactsPlugin.
package savefilesasartifacts

import (
	"bytes"
	"cmp"
	"errors"
	"fmt"
	"maps"
	"math"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

const (
	defaultName = "save_files_as_artifacts_plugin"

	// maxInlineDataSize is the largest file the plugin saves, 20 MiB, the
	// Gemini API's limit for inline data.
	maxInlineDataSize = 20 * 1024 * 1024
)

// modelAccessibleSchemes are the URI schemes the model connectors can read:
// gs:// on Vertex AI, https:// and http:// on hosted endpoints.
var modelAccessibleSchemes = []string{"gs", "https", "http"}

// Config configures the plugin that New returns. The zero value is valid and
// gives adk-python's defaults.
type Config struct {
	// Name is the plugin's name, as returned by (*plugin.Plugin).Name. It also
	// names the session state key "<Name>:pending_delta" in which the plugin
	// keeps the versions it saved until the first agent of the invocation
	// reports them. Empty means "save_files_as_artifacts_plugin".
	Name string

	// DisableFileReference stops the plugin from adding a file-data part after
	// a placeholder. The files are still saved and still reported in the
	// artifact delta. The zero value, false, adds the reference whenever the
	// saved version's canonical URI uses the gs, https or http scheme.
	DisableFileReference bool
}

// New returns a plugin that saves the files a user attaches to a message as
// artifacts of the session, and puts a placeholder naming each file where the
// file was.
//
// The plugin's OnUserMessageCallback handles every part of the message that
// holds inline data, in order, and leaves the other parts where they are. It
// saves the part through the invocation's artifact service under the blob's
// display name, or under "artifact_<invocation ID>_<part index>" when the
// display name is empty, where the index counts all parts of the message. It
// saves a copy, so later edits of the message do not reach the stored
// artifact. A name with the "user:" prefix makes the artifact visible in all
// sessions of the user, and saving a name again adds a version. In place of
// the part it puts the text part [Uploaded Artifact: "<name>"], followed,
// unless [Config.DisableFileReference] is set, by a file-data part that points
// at the saved version when the artifact service gives it a canonical URI with
// the gs, https or http scheme. A blob larger than 20 MiB is not saved and is
// replaced by a text part that names the file, gives its size and asks for a
// smaller file. The callback returns a new message with the original's role
// and never changes the message it is given. When the message has no part
// with inline data, or no part was replaced, it returns nil, so the message
// stays as it was and later plugins' OnUserMessageCallback run.
//
// The versions saved from a message are kept in the session state key
// "<Name>:pending_delta" until the plugin's BeforeAgentCallback, at the first
// agent of the invocation, moves them into the artifact delta of that agent's
// event and sets the key to an empty map. When two parts of one message have
// the same name, the later version is reported.
//
// Without an artifact service the plugin is disabled: OnUserMessageCallback
// returns the message it was given, which stops later plugins'
// OnUserMessageCallback from running for that message, as in adk-python.
// The plugin reports no failure: a file it could not save stays in the
// message as it was, and a file over 20 MiB is replaced by a text part that
// says so. OnUserMessageCallback returns an error, which ends the run, only
// when a save fails because the invocation was canceled or its deadline
// passed, or when the pending key holds a value that is not a map[string]any.
//
// [agent.RunConfig.SaveInputBlobsAsArtifacts] is the runner's older built-in
// way to save uploads. It saves every blob as artifact_<invocation id>_<index>,
// ignores display names, has no size limit, adds no file reference, reports no
// artifact delta, and fails the run when a save fails. adk-python deprecates
// its counterpart in favor of this plugin. When both are enabled, the plugin
// runs first, and the runner saves only the blobs the plugin left in the
// message, that is, those it failed to save.
//
// New returns the error of [plugin.New], which is nil today. The plugin keeps
// no state of its own between calls, so one plugin may serve concurrent
// invocations.
func New(cfg Config) (*plugin.Plugin, error) {
	name := cmp.Or(cfg.Name, defaultName)
	s := &saveFiles{
		pendingKey:          name + ":pending_delta",
		attachFileReference: !cfg.DisableFileReference,
	}
	return plugin.New(plugin.Config{
		Name:                  name,
		OnUserMessageCallback: s.onUserMessage,
		BeforeAgentCallback:   s.beforeAgent,
	})
}

type saveFiles struct {
	pendingKey          string
	attachFileReference bool
}

func (s *saveFiles) onUserMessage(ictx agent.InvocationContext, msg *genai.Content) (*genai.Content, error) {
	arts := ictx.Artifacts()
	if !hasArtifactService(arts) {
		return msg, nil
	}
	if msg == nil || len(msg.Parts) == 0 {
		return nil, nil
	}

	newParts := make([]*genai.Part, 0, len(msg.Parts))
	saved := make(map[string]int64)
	modified := false
	for i, part := range msg.Parts {
		if part == nil || part.InlineData == nil {
			newParts = append(newParts, part)
			continue
		}
		blob := part.InlineData
		name := fileName(blob, ictx.InvocationID(), i)
		if len(blob.Data) > maxInlineDataSize {
			newParts = append(newParts, uploadErrorPart(name, len(blob.Data)))
			modified = true
			continue
		}
		resp, err := arts.Save(ictx, name, savedCopy(part))
		if err != nil {
			// adk-python's `except Exception` does not catch a cancellation.
			if ictx.Err() != nil {
				return nil, fmt.Errorf("savefilesasartifacts: saving part %d: %w", i, err)
			}
			newParts = append(newParts, part)
			continue
		}
		newParts = append(newParts, placeholderPart(name))
		if s.attachFileReference {
			if ref := fileReferencePart(ictx, name, resp.Version, blob.MIMEType); ref != nil {
				newParts = append(newParts, ref)
			}
		}
		saved[name] = resp.Version
		modified = true
	}
	if !modified {
		return nil, nil
	}
	if err := s.mergePending(ictx.Session().State(), saved); err != nil {
		return nil, err
	}
	return &genai.Content{Role: msg.Role, Parts: newParts}, nil
}

// mergePending adds saved to the map under the pending key of st, creating
// the map when the key is missing.
func (s *saveFiles) mergePending(st session.State, saved map[string]int64) error {
	merged := make(map[string]any, len(saved))
	v, err := st.Get(s.pendingKey)
	switch {
	case errors.Is(err, session.ErrStateKeyNotExist):
	case err != nil:
		return fmt.Errorf("savefilesasartifacts: reading session state %q: %w", s.pendingKey, err)
	default:
		m, ok := v.(map[string]any)
		if !ok {
			return fmt.Errorf("savefilesasartifacts: session state %q holds %T, want map[string]any", s.pendingKey, v)
		}
		maps.Copy(merged, m)
	}
	for name, version := range saved {
		merged[name] = version
	}
	if err := st.Set(s.pendingKey, merged); err != nil {
		return fmt.Errorf("savefilesasartifacts: writing session state %q: %w", s.pendingKey, err)
	}
	return nil
}

func (s *saveFiles) beforeAgent(ctx agent.Context) (*genai.Content, error) {
	v, err := ctx.State().Get(s.pendingKey)
	if errors.Is(err, session.ErrStateKeyNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("savefilesasartifacts: reading session state %q: %w", s.pendingKey, err)
	}
	if isEmptyValue(v) {
		return nil, nil
	}
	// A value that is not a map is what makes adk-python's `|=` raise; like
	// adk-python, the plugin then reports nothing and still clears the key.
	var mergeErr error
	if m, ok := v.(map[string]any); ok {
		mergeErr = agent.MergeArtifactDelta(ctx, versionsOf(m))
	}
	setErr := ctx.State().Set(s.pendingKey, map[string]any{})
	if err := errors.Join(mergeErr, setErr); err != nil {
		return nil, fmt.Errorf("savefilesasartifacts: reporting the versions in session state %q: %w", s.pendingKey, err)
	}
	return nil, nil
}

// hasArtifactService reports whether arts reaches an artifact service. A
// hand-built context returns nil Artifacts without one; a context of the
// runner returns an Artifacts whose ArtifactServiceErr method reports it.
func hasArtifactService(arts agent.Artifacts) bool {
	if arts == nil {
		return false
	}
	if a, ok := arts.(interface{ ArtifactServiceErr() error }); ok && a.ArtifactServiceErr() != nil {
		return false
	}
	return true
}

// fileName returns the blob's display name, or the generated name when it is
// empty. index is the part's position among all parts of the message.
func fileName(blob *genai.Blob, invocationID string, index int) string {
	if blob.DisplayName != "" {
		return blob.DisplayName
	}
	return fmt.Sprintf("artifact_%s_%d", invocationID, index)
}

// savedCopy returns a copy of part that shares no blob and no data with it.
// The other fields of the part are shared, as in adk-python's copy.copy.
func savedCopy(part *genai.Part) *genai.Part {
	saved := *part
	blob := *part.InlineData
	blob.Data = bytes.Clone(part.InlineData.Data)
	saved.InlineData = &blob
	return &saved
}

// placeholderPart inserts name verbatim, as adk-python's f-string does; %q
// would escape quotes and non-ASCII characters in it.
func placeholderPart(name string) *genai.Part {
	return genai.NewPartFromText("[Uploaded Artifact: \"" + name + "\"]")
}

func uploadErrorPart(name string, size int) *genai.Part {
	msg := fmt.Sprintf("File %s (%.2f MB) exceeds the maximum supported size of %.0fMB. Please upload a smaller file.",
		name, float64(size)/(1<<20), float64(maxInlineDataSize)/(1<<20))
	return genai.NewPartFromText("[Upload Error: " + msg + "]")
}

// fileReferencePart returns a file-data part for the given version of the
// artifact name, or nil when the version cannot be looked up or its canonical
// URI is not one the model can read.
func fileReferencePart(ictx agent.InvocationContext, name string, version int64, blobMIME string) *genai.Part {
	actx, ok := ictx.(agent.Context)
	if !ok {
		actx = agent.NewContext(ictx)
	}
	rec, err := agent.ArtifactVersion(actx, name, &version)
	if err != nil || rec == nil || !isModelAccessibleURI(rec.CanonicalURI) {
		return nil
	}
	return &genai.Part{FileData: &genai.FileData{
		FileURI:     rec.CanonicalURI,
		MIMEType:    cmp.Or(blobMIME, rec.MimeType),
		DisplayName: name,
	}}
}

// isModelAccessibleURI reports whether the scheme of uri, the text before the
// first ":", is one of modelAccessibleSchemes, ignoring case. The rest of uri
// is not validated: adk-go's GCS service writes file names into its URIs
// unescaped, so a name such as "50%off.pdf" would fail url.Parse.
func isModelAccessibleURI(uri string) bool {
	scheme, _, ok := strings.Cut(uri, ":")
	if !ok {
		return false
	}
	for _, s := range modelAccessibleSchemes {
		if strings.EqualFold(scheme, s) {
			return true
		}
	}
	return false
}

// isEmptyValue reports whether v is a value a session can hold that Python's
// truth test treats as false.
func isEmptyValue(v any) bool {
	switch x := v.(type) {
	case nil:
		return true
	case bool:
		return !x
	case int:
		return x == 0
	case int64:
		return x == 0
	case float64:
		return x == 0
	case string:
		return x == ""
	case []any:
		return len(x) == 0
	case map[string]any:
		return len(x) == 0
	}
	return false
}

// versionsOf returns the entries of m that hold an integral number. A session
// that went through JSON holds numbers as float64. Other entries are left out,
// because an artifact delta maps names to int64 versions.
func versionsOf(m map[string]any) map[string]int64 {
	versions := make(map[string]int64, len(m))
	for name, v := range m {
		switch x := v.(type) {
		case int64:
			versions[name] = x
		case int:
			versions[name] = int64(x)
		case float64:
			if x == math.Trunc(x) && math.Abs(x) < math.MaxInt64 {
				versions[name] = int64(x)
			}
		}
	}
	return versions
}
