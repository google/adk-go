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
	"math"
	"slices"
	"strings"
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
					Description: "Optional revision number to load for each artifact in artifact_names. Revisions are numbered from 0. If omitted, the latest revision is loaded.",
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
								Description: "Revision number to load. Revisions are numbered from 0. If omitted, the latest revision is loaded.",
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

// artifactVersionSelector names one artifact revision. A nil Version selects
// the latest revision.
type artifactVersionSelector struct {
	Name    string
	Version *int64
}

// loadItem is one artifact a load_artifacts response asks for. position names
// where the request listed it, such as "artifact_versions[1]", so an error can
// identify the item without echoing the model-supplied name.
type loadItem struct {
	artifactVersionSelector
	position string
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

	if rawVersion, exists := m["version"]; exists && rawVersion != nil {
		version, err := parseNonNegativeInt(rawVersion, "version")
		if err != nil {
			return nil, err
		}
		result["version"] = version
	}

	if rawSelectors, exists := m["artifact_versions"]; exists && rawSelectors != nil {
		selectors, err := parseArtifactVersionSelectors(rawSelectors)
		if err != nil {
			return nil, err
		}
		if len(selectors) > 0 {
			encoded := make([]map[string]any, len(selectors))
			for i, sel := range selectors {
				encoded[i] = map[string]any{"name": sel.Name}
				if sel.Version != nil {
					encoded[i]["version"] = int(*sel.Version)
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
		return fmt.Errorf("failed to list artifacts: %w", redact(err))
	}
	if err := appendInitialInstructions(req, listed.FileNames); err != nil {
		return err
	}
	return t.processLoadArtifactsFunctionCall(ctx, req)
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

func (t *artifactsTool) processLoadArtifactsFunctionCall(ctx agent.Context, req *model.LLMRequest) error {
	if len(req.Contents) == 0 {
		return nil
	}
	responses := slices.DeleteFunc(utils.FunctionResponses(req.Contents[len(req.Contents)-1]), func(response *genai.FunctionResponse) bool {
		return response.Name != t.Name()
	})
	// Like Python, ignore a malformed sibling response when multiple
	// load_artifacts calls run in parallel, while preserving single-call errors.
	parallel := len(responses) > 1
	var jobs []loadJob
	for i, response := range responses {
		prefix := ""
		if parallel {
			prefix = fmt.Sprintf("load_artifacts[%d].", i)
		}
		responseJobs, err := loadJobsFromResponse(response.Response, prefix)
		if err != nil {
			if parallel {
				continue
			}
			return err
		}
		jobs = append(jobs, responseJobs...)
	}
	if len(jobs) == 0 {
		return nil
	}

	// Deliberate divergence from adk-python (load_artifacts_tool.py:553-558,
	// 582-584), which logs a warning and silently skips unavailable or missing
	// artifacts: Go library code has no global logger, and reporting
	// model-fixable name errors (fs.ErrNotExist or fs.ErrInvalid from the
	// service) back to the model as user-role text lets it recover while still
	// letting custom services accept nested names. Storage, permission,
	// unsupported, and cancellation errors still fail the run.
	artifactsService := ctx.Artifacts()
	results := make([]*genai.Content, len(jobs))
	notes := make([]string, len(jobs))
	group, childCtx := errgroup.WithContext(ctx)
	for i, job := range jobs {
		group.Go(func() error {
			var content *genai.Content
			var err error
			if job.listVersions {
				content, err = t.listArtifactVersions(childCtx, artifactsService, job.loadItem)
			} else {
				content, err = t.loadIndividualArtifact(childCtx, artifactsService, job.loadItem)
			}
			if err != nil {
				if note := modelFixableNote(job, err); note != "" {
					notes[i] = note
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
	if failureNotes := slices.DeleteFunc(notes, func(note string) bool { return note == "" }); len(failureNotes) > 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText(strings.Join(failureNotes, "\n"), genai.RoleUser))
	}
	return nil
}

// modelFixableNote formats a user-role note for errors the model can correct on
// its next turn (missing artifact/version or invalid filename). It returns ""
// for errors that must continue to fail the run.
func modelFixableNote(job loadJob, err error) string {
	switch {
	case errors.Is(err, fs.ErrNotExist):
		if !job.listVersions && job.Version != nil {
			return fmt.Sprintf("artifact %q (version %d) was not found", job.Name, *job.Version)
		}
		return fmt.Sprintf("artifact %q was not found", job.Name)
	case errors.Is(err, fs.ErrInvalid):
		return fmt.Sprintf(
			"artifact %q is not a valid artifact name; names must be non-empty plain filenames without path separators",
			job.Name,
		)
	default:
		return ""
	}
}

// loadJob is one artifact to load, or whose versions to list.
type loadJob struct {
	loadItem
	listVersions bool
}

// loadJobsFromResponse returns the work one load_artifacts function response
// asks for. prefix goes in front of each item's position, so items from
// parallel responses can be told apart. When list_versions is set, the
// response lists the versions of its names and loads nothing.
func loadJobsFromResponse(response map[string]any, prefix string) ([]loadJob, error) {
	var artifactNames []string
	if raw, ok := response["artifact_names"]; ok && raw != nil {
		var err error
		artifactNames, err = artifactNamesFromResponse(raw)
		if err != nil {
			return nil, err
		}
	}

	var version *int64
	if rawVersion, ok := response["version"]; ok && rawVersion != nil {
		v, err := parseNonNegativeInt(rawVersion, "version")
		if err != nil {
			return nil, err
		}
		version = new(int64(v))
	}

	var selectors []artifactVersionSelector
	if rawSelectors, ok := response["artifact_versions"]; ok && rawSelectors != nil {
		var err error
		selectors, err = parseArtifactVersionSelectors(rawSelectors)
		if err != nil {
			return nil, err
		}
	}

	nameJobs := make([]loadJob, len(artifactNames))
	for i, name := range artifactNames {
		nameJobs[i] = loadJob{loadItem: loadItem{
			artifactVersionSelector: artifactVersionSelector{Name: name, Version: version},
			position:                fmt.Sprintf("%sartifact_names[%d]", prefix, i),
		}}
	}

	if rawListVersions, ok := response["list_versions"]; ok && rawListVersions != nil {
		listVersions, extraNames, err := parseListVersionsArg(rawListVersions)
		if err != nil {
			return nil, err
		}
		for i, name := range extraNames {
			if !slices.Contains(artifactNames, name) {
				artifactNames = append(artifactNames, name)
				nameJobs = append(nameJobs, loadJob{loadItem: loadItem{
					artifactVersionSelector: artifactVersionSelector{Name: name, Version: version},
					position:                fmt.Sprintf("%slist_versions[%d]", prefix, i),
				}})
			}
		}
		if listVersions {
			for i := range nameJobs {
				nameJobs[i].listVersions = true
			}
			return nameJobs, nil
		}
	}

	jobs := nameJobs
	for i, sel := range selectors {
		jobs = append(jobs, loadJob{loadItem: loadItem{
			artifactVersionSelector: sel,
			position:                fmt.Sprintf("%sartifact_versions[%d]", prefix, i),
		}})
	}
	return jobs, nil
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

// redactFor redacts err, an error the artifact service returned for name. When
// err wraps none of redactSentinels and name is empty or contains a path
// separator, the message adds that fact to the error type. The type stays in
// the message because a custom service that accepts such names can fail for
// storage reasons, and the name is checked only after the service has failed,
// so such a service still serves them.
func redactFor(name string, err error) error {
	if r := redactSentinel(err); r != nil {
		return r
	}
	r := redactType(err)
	switch {
	case name == "":
		r.msg += "; the name is empty"
	case strings.ContainsAny(name, `/\`):
		r.msg += "; the name contains a path separator, which the in-memory and GCS artifact services reject"
	}
	return r
}

func (t *artifactsTool) loadIndividualArtifact(ctx context.Context, artifactsService agent.Artifacts, item loadItem) (*genai.Content, error) {
	artifactName, version := item.Name, item.Version
	if version != nil {
		resp, err := artifactsService.LoadVersion(ctx, artifactName, version)
		if err != nil {
			return nil, fmt.Errorf("failed to load artifact at %s (version %d): %w", item.position, *version, redactFor(artifactName, err))
		}
		return &genai.Content{
			Parts: []*genai.Part{
				genai.NewPartFromText(fmt.Sprintf("Artifact %s (version %d) is:", artifactName, *version)),
				resp.Part,
			},
			Role: genai.RoleUser,
		}, nil
	}
	resp, err := artifactsService.Load(ctx, artifactName)
	if err != nil {
		return nil, fmt.Errorf("failed to load artifact at %s: %w", item.position, redactFor(artifactName, err))
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

func (t *artifactsTool) listArtifactVersions(ctx context.Context, artifactsService agent.Artifacts, item loadItem) (*genai.Content, error) {
	artifactName := item.Name
	versionsResp, err := artifactinternal.Versions(ctx, artifactsService, artifactName)
	if err != nil {
		return nil, fmt.Errorf("failed to list versions of artifact at %s: %w", item.position, redactFor(artifactName, err))
	}
	versions := slices.Clone(versionsResp.Versions)
	slices.Sort(versions)

	summaries := make([]artifactVersionSummary, 0, len(versions))
	for _, v := range versions {
		metaResp, err := artifactinternal.GetArtifactVersion(ctx, artifactsService, artifactName, v)
		if err != nil {
			return nil, fmt.Errorf("failed to get version %d metadata for artifact at %s: %w", v, item.position, redactFor(artifactName, err))
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
		return nil, fmt.Errorf("failed to marshal version metadata for artifact at %s: %w", item.position, err)
	}
	return &genai.Content{
		Parts: []*genai.Part{
			genai.NewPartFromText(fmt.Sprintf("Versions of artifact %s:", artifactName)),
			genai.NewPartFromText(string(encoded)),
		},
		Role: genai.RoleUser,
	}, nil
}

// redactedError reports msg while keeping cause reachable through errors.Is and
// errors.As. Wrapping with %w would put the cause's own text in the message,
// which is the thing being redacted.
type redactedError struct {
	msg   string
	cause error
}

func (e *redactedError) Error() string { return e.msg }
func (e *redactedError) Unwrap() error { return e.cause }

// redactSentinels are the errors whose text redact keeps, checked in order.
var redactSentinels = []error{
	fs.ErrNotExist,
	fs.ErrPermission,
	errors.ErrUnsupported,
	context.Canceled,
	context.DeadlineExceeded,
}

// redact wraps err, an error from the artifact service, in an error whose text
// is that of the first error in redactSentinels that err wraps. When err wraps
// none of them, the text is the type of the innermost error in its chain, so a
// service error that wraps no sentinel loses its own message. errors.Is and
// errors.As still reach err. Services write the object path or the request URL
// into their errors, and both carry the model-supplied artifact name and the
// user and session IDs.
func redact(err error) error {
	if r := redactSentinel(err); r != nil {
		return r
	}
	return redactType(err)
}

// redactSentinel returns err redacted to the text of the first error in
// redactSentinels it wraps, or nil if it wraps none.
func redactSentinel(err error) error {
	for _, sentinel := range redactSentinels {
		if errors.Is(err, sentinel) {
			return &redactedError{msg: sentinel.Error(), cause: err}
		}
	}
	return nil
}

// redactType returns err redacted to the type of the innermost error in its
// chain.
func redactType(err error) *redactedError {
	inner := err
	for errors.Unwrap(inner) != nil {
		inner = errors.Unwrap(inner)
	}
	return &redactedError{msg: fmt.Sprintf("artifact service error of type %T", inner), cause: err}
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
		var version *int64
		if verRaw, exists := obj["version"]; exists && verRaw != nil {
			v, err := parseNonNegativeInt(verRaw, fmt.Sprintf("artifact_versions[%d].version", i))
			if err != nil {
				return nil, err
			}
			version = new(int64(v))
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
