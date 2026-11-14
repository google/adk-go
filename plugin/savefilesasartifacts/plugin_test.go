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

package savefilesasartifacts_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"iter"
	"maps"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/savefilesasartifacts"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

const (
	appName    = "app"
	userID     = "user"
	sessionID  = "s"
	defaultKey = "save_files_as_artifacts_plugin:pending_delta"
)

// fakeService is an artifact.Service double. Save records each request, with
// the part it was handed and no copy, and answers with saveFn, or with
// version 1 when saveFn is nil. GetArtifactVersion records each request and
// answers with lookupFn, or with an error when lookupFn is nil. The embedded
// nil Service panics on every other method, which the plugin must not call.
type fakeService struct {
	artifact.Service
	saveFn   func(ctx context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error)
	lookupFn func(req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error)

	mu      sync.Mutex
	saves   []*artifact.SaveRequest
	lookups []*artifact.GetArtifactVersionRequest
}

func (f *fakeService) Save(ctx context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	f.mu.Lock()
	f.saves = append(f.saves, req)
	f.mu.Unlock()
	if f.saveFn != nil {
		return f.saveFn(ctx, req)
	}
	return &artifact.SaveResponse{Version: 1}, nil
}

func (f *fakeService) GetArtifactVersion(_ context.Context, req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	f.mu.Lock()
	f.lookups = append(f.lookups, req)
	f.mu.Unlock()
	if f.lookupFn == nil {
		return nil, errors.New("unexpected version lookup")
	}
	return f.lookupFn(req)
}

func (f *fakeService) savedNames() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var names []string
	for _, req := range f.saves {
		names = append(names, req.FileName)
	}
	return names
}

// uriLookup answers a version lookup with "<prefix>/<file name>/versions/<version>"
// and mime as the record's MIME type.
func uriLookup(prefix, mime string) func(*artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	return func(req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
		return &artifact.GetArtifactVersionResponse{ArtifactVersion: &artifact.ArtifactVersion{
			Version:        req.Version,
			CanonicalURI:   fmt.Sprintf("%s/%s/versions/%d", prefix, req.FileName, req.Version),
			CustomMetadata: map[string]any{},
			MimeType:       mime,
		}}, nil
	}
}

// recordLookup answers every version lookup with rec.
func recordLookup(rec *artifact.ArtifactVersion) func(*artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	return func(*artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
		return &artifact.GetArtifactVersionResponse{ArtifactVersion: rec}, nil
	}
}

func blobPart(name string, data []byte, mime string) *genai.Part {
	return &genai.Part{InlineData: &genai.Blob{DisplayName: name, Data: data, MIMEType: mime}}
}

func userMessage(parts ...*genai.Part) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: parts}
}

func newPlugin(t *testing.T, cfg savefilesasartifacts.Config) *plugin.Plugin {
	t.Helper()
	p, err := savefilesasartifacts.New(cfg)
	if err != nil {
		t.Fatalf("New(%+v) error = %v", cfg, err)
	}
	return p
}

// testApp is a runner over an in-memory session service whose root agent,
// root_agent, records the UserContent it sees and yields one "reply" event.
type testApp struct {
	runner   *runner.Runner
	sessions session.Service

	mu           sync.Mutex
	userContents map[string][]*genai.Content // per session ID, one per turn
}

// newTestApp returns a testApp with svc as the runner's artifact service (nil
// for none) and plugins in that order.
func newTestApp(t *testing.T, svc artifact.Service, plugins ...*plugin.Plugin) *testApp {
	t.Helper()
	a := &testApp{sessions: session.InMemoryService(), userContents: map[string][]*genai.Content{}}
	root, err := agent.New(agent.Config{
		Name: "root_agent",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				a.mu.Lock()
				a.userContents[ic.Session().ID()] = append(a.userContents[ic.Session().ID()], ic.UserContent())
				a.mu.Unlock()
				ev := session.NewEvent(ic, ic.InvocationID())
				ev.Author = "root_agent"
				ev.Content = genai.NewContentFromText("reply", genai.RoleModel)
				yield(ev, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	a.runner, err = runner.New(runner.Config{
		AppName:         appName,
		Agent:           root,
		SessionService:  a.sessions,
		ArtifactService: svc,
		PluginConfig:    runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	return a
}

func (a *testApp) createSession(t *testing.T, id string, state map[string]any) {
	t.Helper()
	if _, err := a.sessions.Create(t.Context(), &session.CreateRequest{AppName: appName, UserID: userID, SessionID: id, State: state}); err != nil {
		t.Fatalf("Create(%q) error = %v", id, err)
	}
}

// run runs one turn and returns the events it yielded up to the first error,
// and that error.
func (a *testApp) run(ctx context.Context, id string, msg *genai.Content, opts ...runner.RunOption) ([]*session.Event, error) {
	var events []*session.Event
	for ev, err := range a.runner.Run(ctx, userID, id, msg, agent.RunConfig{}, opts...) {
		if err != nil {
			return events, err
		}
		events = append(events, ev)
	}
	return events, nil
}

// mustRun is run for a turn that must succeed and yield at least one event.
func (a *testApp) mustRun(t *testing.T, id string, msg *genai.Content) []*session.Event {
	t.Helper()
	events, err := a.run(t.Context(), id, msg)
	if err != nil {
		t.Fatalf("Run(%q) error = %v", id, err)
	}
	if len(events) == 0 {
		t.Fatalf("Run(%q) yielded no events", id)
	}
	return events
}

func (a *testApp) storedSession(t *testing.T, id string) session.Session {
	t.Helper()
	resp, err := a.sessions.Get(t.Context(), &session.GetRequest{AppName: appName, UserID: userID, SessionID: id})
	if err != nil {
		t.Fatalf("Get(%q) error = %v", id, err)
	}
	return resp.Session
}

func (a *testApp) storedState(t *testing.T, id string) map[string]any {
	t.Helper()
	return maps.Collect(a.storedSession(t, id).State().All())
}

// storedUserContents returns the contents of the session's user events, in
// order.
func (a *testApp) storedUserContents(t *testing.T, id string) []*genai.Content {
	t.Helper()
	var contents []*genai.Content
	for ev := range a.storedSession(t, id).Events().All() {
		if ev.Author == genai.RoleUser {
			contents = append(contents, ev.Content)
		}
	}
	return contents
}

// directCall describes the runner turn on whose invocation context
// callOnUserMessage calls a plugin's OnUserMessageCallback.
type directCall struct {
	svc          artifact.Service // the runner's artifact service; nil for none
	state        map[string]any   // the session's initial state
	invocationID string           // the turn's invocation ID; "" lets the runner mint one
	ctx          context.Context  // the turn's context; nil means t.Context()
	// wrap, when set, replaces the runner's invocation context with the one
	// it returns before the callback is called.
	wrap func(agent.InvocationContext) agent.InvocationContext
}

type callResult struct {
	out   *genai.Content
	err   error
	state map[string]any // the session state right after the callback returned
}

// callOnUserMessage runs one turn of a runner built as c describes, whose only
// plugin hands msg to p's OnUserMessageCallback on the invocation context the
// runner gives it, and records what that returned. The runner's own message
// is a text part, so msg may be anything, nil included.
func callOnUserMessage(t *testing.T, p *plugin.Plugin, c directCall, msg *genai.Content) callResult {
	t.Helper()
	var (
		res    callResult
		called bool
	)
	caller, err := plugin.New(plugin.Config{
		Name: "caller",
		OnUserMessageCallback: func(ictx agent.InvocationContext, _ *genai.Content) (*genai.Content, error) {
			called = true
			if c.wrap != nil {
				ictx = c.wrap(ictx)
			}
			res.out, res.err = p.OnUserMessageCallback()(ictx, msg)
			res.state = maps.Collect(ictx.Session().State().All())
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	a := newTestApp(t, c.svc, caller)
	a.createSession(t, sessionID, c.state)
	ctx := c.ctx
	if ctx == nil {
		ctx = t.Context()
	}
	var opts []runner.RunOption
	if c.invocationID != "" {
		opts = append(opts, runner.WithInvocationID(c.invocationID))
	}
	// Only the callback is under test; the rest of the turn may fail, for
	// example after the callback canceled ctx.
	_, runErr := a.run(ctx, sessionID, genai.NewContentFromText("hi", genai.RoleUser), opts...)
	if !called {
		t.Fatalf("OnUserMessageCallback was not called; Run error = %v", runErr)
	}
	return res
}

func TestNew_DefaultAndCustomName(t *testing.T) {
	if got := newPlugin(t, savefilesasartifacts.Config{}).Name(); got != "save_files_as_artifacts_plugin" {
		t.Errorf("New(Config{}).Name() = %q, want %q", got, "save_files_as_artifacts_plugin")
	}

	custom := newPlugin(t, savefilesasartifacts.Config{Name: "custom_saver"})
	if got := custom.Name(); got != "custom_saver" {
		t.Errorf("New(Config{Name: custom_saver}).Name() = %q, want %q", got, "custom_saver")
	}
	msg := userMessage(blobPart("a.txt", []byte("a"), "text/plain"))

	res := callOnUserMessage(t, custom, directCall{svc: artifact.InMemoryService()}, msg)
	if res.err != nil {
		t.Fatalf("OnUserMessageCallback() error = %v", res.err)
	}
	if diff := cmp.Diff(map[string]any{"a.txt": int64(1)}, res.state["custom_saver:pending_delta"]); diff != "" {
		t.Errorf("state[custom_saver:pending_delta] after OnUserMessageCallback mismatch (-want +got):\n%s", diff)
	}
	if v, ok := res.state[defaultKey]; ok {
		t.Errorf("state[%s] = %v, want no such key", defaultKey, v)
	}

	a := newTestApp(t, artifact.InMemoryService(), custom)
	a.createSession(t, sessionID, nil)
	events := a.mustRun(t, sessionID, msg)
	if diff := cmp.Diff(map[string]int64{"a.txt": 1}, events[0].Actions.ArtifactDelta); diff != "" {
		t.Errorf("first event ArtifactDelta mismatch (-want +got):\n%s", diff)
	}
	state := a.storedState(t, sessionID)
	if diff := cmp.Diff(map[string]any{}, state["custom_saver:pending_delta"]); diff != "" {
		t.Errorf("stored state[custom_saver:pending_delta] mismatch (-want +got):\n%s", diff)
	}
	if v, ok := state[defaultKey]; ok {
		t.Errorf("stored state[%s] = %v, want no such key", defaultKey, v)
	}
}

func TestRunner_EndToEnd_ArtifactDeltaAndPlaceholder(t *testing.T) {
	const quoted = `test file with "quotes" and spaces.pdf`
	svc := artifact.InMemoryService()
	a := newTestApp(t, svc, newPlugin(t, savefilesasartifacts.Config{}))
	a.createSession(t, sessionID, nil)

	turn1 := a.mustRun(t, sessionID, userMessage(blobPart(quoted, []byte("pdf-1"), "application/pdf")))

	placeholder1 := []*genai.Part{{Text: `[Uploaded Artifact: "test file with "quotes" and spaces.pdf"]`}}
	if got := turn1[0].Author; got != "root_agent" {
		t.Errorf("turn 1 first event Author = %q, want %q", got, "root_agent")
	}
	if diff := cmp.Diff(map[string]int64{quoted: 1}, turn1[0].Actions.ArtifactDelta); diff != "" {
		t.Errorf("turn 1 first event ArtifactDelta mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{defaultKey: map[string]any{}}, turn1[0].Actions.StateDelta); diff != "" {
		t.Errorf("turn 1 first event StateDelta mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(placeholder1, a.userContents[sessionID][0].Parts); diff != "" {
		t.Errorf("turn 1 UserContent parts mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(placeholder1, a.storedUserContents(t, sessionID)[0].Parts); diff != "" {
		t.Errorf("turn 1 stored user event parts mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{}, a.storedState(t, sessionID)[defaultKey]); diff != "" {
		t.Errorf("stored state after turn 1 mismatch (-want +got):\n%s", diff)
	}

	turn2 := a.mustRun(t, sessionID, userMessage(blobPart("blob_2.pdf", []byte("pdf-2"), "application/pdf")))

	if diff := cmp.Diff(map[string]int64{"blob_2.pdf": 1}, turn2[0].Actions.ArtifactDelta); diff != "" {
		t.Errorf("turn 2 first event ArtifactDelta mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{defaultKey: map[string]any{}}, turn2[0].Actions.StateDelta); diff != "" {
		t.Errorf("turn 2 first event StateDelta mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{}, a.storedState(t, sessionID)[defaultKey]); diff != "" {
		t.Errorf("stored state after turn 2 mismatch (-want +got):\n%s", diff)
	}

	loaded, err := svc.Load(t.Context(), &artifact.LoadRequest{AppName: appName, UserID: userID, SessionID: sessionID, FileName: quoted})
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	want := &genai.Part{InlineData: &genai.Blob{DisplayName: quoted, Data: []byte("pdf-1"), MIMEType: "application/pdf"}}
	if diff := cmp.Diff(want, loaded.Part); diff != "" {
		t.Errorf("loaded artifact mismatch (-want +got):\n%s", diff)
	}
}

func TestOnUserMessage_GeneratedFileNameUsesPartIndexAndMultipleFiles(t *testing.T) {
	svc := &fakeService{lookupFn: uriLookup("gs://bucket", "application/octet-stream")}
	msg := userMessage(
		genai.NewPartFromText("intro"),
		blobPart("", []byte("b1"), "text/plain"),
		blobPart("second.jpg", []byte("b2"), "image/jpeg"),
	)

	res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{}), directCall{svc: svc, invocationID: "inv_42"}, msg)

	if res.err != nil {
		t.Fatalf("OnUserMessageCallback() error = %v", res.err)
	}
	if diff := cmp.Diff([]string{"artifact_inv_42_1", "second.jpg"}, svc.savedNames()); diff != "" {
		t.Errorf("saved names mismatch (-want +got):\n%s", diff)
	}
	want := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: "intro"},
		{Text: `[Uploaded Artifact: "artifact_inv_42_1"]`},
		{FileData: &genai.FileData{FileURI: "gs://bucket/artifact_inv_42_1/versions/1", MIMEType: "text/plain", DisplayName: "artifact_inv_42_1"}},
		{Text: `[Uploaded Artifact: "second.jpg"]`},
		{FileData: &genai.FileData{FileURI: "gs://bucket/second.jpg/versions/1", MIMEType: "image/jpeg", DisplayName: "second.jpg"}},
	}}
	if diff := cmp.Diff(want, res.out); diff != "" {
		t.Errorf("OnUserMessageCallback() content mismatch (-want +got):\n%s", diff)
	}
}

// The first blob becomes two parts (placeholder and file reference), so the
// generated name of the second blob tells the part index from the number of
// parts built so far.
func TestOnUserMessage_GeneratedFileNameIndexCountsMessagePartsNotOutputParts(t *testing.T) {
	svc := &fakeService{lookupFn: uriLookup("gs://bucket", "application/octet-stream")}
	msg := userMessage(
		blobPart("a.pdf", []byte("a"), "application/pdf"),
		blobPart("", []byte("b"), "text/plain"),
	)

	res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{}), directCall{svc: svc, invocationID: "inv_42"}, msg)

	if res.err != nil {
		t.Fatalf("OnUserMessageCallback() error = %v", res.err)
	}
	if diff := cmp.Diff([]string{"a.pdf", "artifact_inv_42_1"}, svc.savedNames()); diff != "" {
		t.Errorf("saved names mismatch (-want +got):\n%s", diff)
	}
	want := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		{Text: `[Uploaded Artifact: "a.pdf"]`},
		{FileData: &genai.FileData{FileURI: "gs://bucket/a.pdf/versions/1", MIMEType: "application/pdf", DisplayName: "a.pdf"}},
		{Text: `[Uploaded Artifact: "artifact_inv_42_1"]`},
		{FileData: &genai.FileData{FileURI: "gs://bucket/artifact_inv_42_1/versions/1", MIMEType: "text/plain", DisplayName: "artifact_inv_42_1"}},
	}}
	if diff := cmp.Diff(want, res.out); diff != "" {
		t.Errorf("OnUserMessageCallback() content mismatch (-want +got):\n%s", diff)
	}
}

func TestOnUserMessage_FileReferenceSchemesAndMIMEFallback(t *testing.T) {
	placeholder := &genai.Part{Text: `[Uploaded Artifact: "doc.pdf"]`}
	tests := []struct {
		name      string
		blobMIME  string
		lookup    func(*artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error)
		wantParts []*genai.Part
	}{
		{
			name:      "gs keeps the blob's MIME type",
			blobMIME:  "application/pdf",
			lookup:    recordLookup(&artifact.ArtifactVersion{Version: 3, CanonicalURI: "gs://b/f/3", MimeType: "application/octet-stream"}),
			wantParts: []*genai.Part{placeholder, {FileData: &genai.FileData{FileURI: "gs://b/f/3", MIMEType: "application/pdf", DisplayName: "doc.pdf"}}},
		},
		{
			name:      "upper-case https falls back to the record's MIME type",
			blobMIME:  "",
			lookup:    recordLookup(&artifact.ArtifactVersion{Version: 3, CanonicalURI: "HTTPS://example.com/f/3", MimeType: "image/png"}),
			wantParts: []*genai.Part{placeholder, {FileData: &genai.FileData{FileURI: "HTTPS://example.com/f/3", MIMEType: "image/png", DisplayName: "doc.pdf"}}},
		},
		{
			name:      "http",
			blobMIME:  "text/csv",
			lookup:    recordLookup(&artifact.ArtifactVersion{Version: 3, CanonicalURI: "http://example.com/f/3"}),
			wantParts: []*genai.Part{placeholder, {FileData: &genai.FileData{FileURI: "http://example.com/f/3", MIMEType: "text/csv", DisplayName: "doc.pdf"}}},
		},
		{
			name:      "memory scheme",
			blobMIME:  "application/pdf",
			lookup:    recordLookup(&artifact.ArtifactVersion{Version: 3, CanonicalURI: "memory://app/u/s/f/3"}),
			wantParts: []*genai.Part{placeholder},
		},
		{
			name:      "s3 scheme",
			blobMIME:  "application/pdf",
			lookup:    recordLookup(&artifact.ArtifactVersion{Version: 3, CanonicalURI: "s3://bucket/f/3"}),
			wantParts: []*genai.Part{placeholder},
		},
		{
			name:      "empty scheme",
			blobMIME:  "application/pdf",
			lookup:    recordLookup(&artifact.ArtifactVersion{Version: 3, CanonicalURI: "://bad-uri"}),
			wantParts: []*genai.Part{placeholder},
		},
		{
			name:      "no scheme separator",
			blobMIME:  "application/pdf",
			lookup:    recordLookup(&artifact.ArtifactVersion{Version: 3, CanonicalURI: "https"}),
			wantParts: []*genai.Part{placeholder},
		},
		{
			name:      "empty URI",
			blobMIME:  "application/pdf",
			lookup:    recordLookup(&artifact.ArtifactVersion{Version: 3, CanonicalURI: ""}),
			wantParts: []*genai.Part{placeholder},
		},
		{
			name:     "lookup error",
			blobMIME: "application/pdf",
			lookup: func(*artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
				return nil, errors.New("lookup failed")
			},
			wantParts: []*genai.Part{placeholder},
		},
		{
			name:      "nil record",
			blobMIME:  "application/pdf",
			lookup:    recordLookup(nil),
			wantParts: []*genai.Part{placeholder},
		},
		{
			name:      "gs URI with a percent sign that is not an escape",
			blobMIME:  "application/pdf",
			lookup:    recordLookup(&artifact.ArtifactVersion{Version: 3, CanonicalURI: "gs://b/50%off.pdf/3"}),
			wantParts: []*genai.Part{placeholder, {FileData: &genai.FileData{FileURI: "gs://b/50%off.pdf/3", MIMEType: "application/pdf", DisplayName: "doc.pdf"}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := &fakeService{
				saveFn: func(context.Context, *artifact.SaveRequest) (*artifact.SaveResponse, error) {
					return &artifact.SaveResponse{Version: 3}, nil
				},
				lookupFn: tt.lookup,
			}
			msg := userMessage(blobPart("doc.pdf", []byte("%PDF"), tt.blobMIME))

			res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{}), directCall{svc: svc}, msg)

			if res.err != nil {
				t.Fatalf("OnUserMessageCallback() error = %v", res.err)
			}
			if res.out == nil {
				t.Fatal("OnUserMessageCallback() content = nil, want the replacement")
			}
			if diff := cmp.Diff(tt.wantParts, res.out.Parts); diff != "" {
				t.Errorf("parts mismatch (-want +got):\n%s", diff)
			}
			wantLookups := []*artifact.GetArtifactVersionRequest{{AppName: appName, UserID: userID, SessionID: sessionID, FileName: "doc.pdf", Version: 3}}
			if diff := cmp.Diff(wantLookups, svc.lookups); diff != "" {
				t.Errorf("version lookups mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(map[string]any{"doc.pdf": int64(3)}, res.state[defaultKey]); diff != "" {
				t.Errorf("pending delta mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// versionArtifacts is the agent.Artifacts of a hand-built invocation context
// that can also look up artifact versions. Save answers version 4, and
// ArtifactVersion answers a gs:// record for the version it is asked for and
// records the request. The embedded nil Artifacts panics on every other
// method, which the plugin must not call.
type versionArtifacts struct {
	agent.Artifacts
	lookups []string
}

func (a *versionArtifacts) Save(context.Context, string, *genai.Part) (*artifact.SaveResponse, error) {
	return &artifact.SaveResponse{Version: 4}, nil
}

func (a *versionArtifacts) ArtifactVersion(_ context.Context, name string, version *int64) (*artifact.ArtifactVersion, error) {
	if version == nil {
		return nil, errors.New("unexpected lookup of the latest version")
	}
	a.lookups = append(a.lookups, fmt.Sprintf("%s@%d", name, *version))
	return &artifact.ArtifactVersion{Version: *version, CanonicalURI: fmt.Sprintf("gs://b/%s/%d", name, *version), MimeType: "application/octet-stream"}, nil
}

// handBuiltContext is an agent.InvocationContext that is not an agent.Context,
// such as an application or a test harness builds, whose Artifacts is arts.
type handBuiltContext struct {
	agent.InvocationContext
	arts agent.Artifacts
}

func (c handBuiltContext) Artifacts() agent.Artifacts { return c.arts }

func TestOnUserMessage_FileReferenceOnHandBuiltInvocationContext(t *testing.T) {
	arts := &versionArtifacts{}
	var wrapped agent.InvocationContext
	wrap := func(ictx agent.InvocationContext) agent.InvocationContext {
		wrapped = handBuiltContext{InvocationContext: ictx, arts: arts}
		return wrapped
	}
	msg := userMessage(blobPart("doc.pdf", []byte("%PDF"), "application/pdf"))

	res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{}), directCall{wrap: wrap}, msg)

	if _, ok := wrapped.(agent.Context); ok {
		t.Fatal("the hand-built invocation context is an agent.Context; the test needs one that is not")
	}
	if res.err != nil {
		t.Fatalf("OnUserMessageCallback() error = %v", res.err)
	}
	want := userMessage(
		&genai.Part{Text: `[Uploaded Artifact: "doc.pdf"]`},
		&genai.Part{FileData: &genai.FileData{FileURI: "gs://b/doc.pdf/4", MIMEType: "application/pdf", DisplayName: "doc.pdf"}},
	)
	if diff := cmp.Diff(want, res.out); diff != "" {
		t.Errorf("OnUserMessageCallback() content mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"doc.pdf@4"}, arts.lookups); diff != "" {
		t.Errorf("version lookups mismatch (-want +got):\n%s", diff)
	}
}

func TestDisableFileReference_StillReportsArtifactDelta(t *testing.T) {
	svc := &fakeService{lookupFn: uriLookup("gs://bucket", "application/pdf")}
	a := newTestApp(t, svc, newPlugin(t, savefilesasartifacts.Config{DisableFileReference: true}))
	a.createSession(t, sessionID, nil)

	events := a.mustRun(t, sessionID, userMessage(blobPart("doc.pdf", []byte("%PDF"), "application/pdf")))

	if diff := cmp.Diff([]*genai.Part{{Text: `[Uploaded Artifact: "doc.pdf"]`}}, a.userContents[sessionID][0].Parts); diff != "" {
		t.Errorf("UserContent parts mismatch (-want +got):\n%s", diff)
	}
	if len(svc.lookups) != 0 {
		t.Errorf("version lookups = %d, want 0", len(svc.lookups))
	}
	if diff := cmp.Diff(map[string]int64{"doc.pdf": 1}, events[0].Actions.ArtifactDelta); diff != "" {
		t.Errorf("first event ArtifactDelta mismatch (-want +got):\n%s", diff)
	}
}

func TestOnUserMessage_EmptyAndNilMessage(t *testing.T) {
	tests := []struct {
		name string
		msg  *genai.Content
	}{
		{name: "empty parts", msg: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{}}},
		{name: "nil parts", msg: &genai.Content{Role: genai.RoleUser}},
		{name: "nil message", msg: nil},
		{name: "nil part only", msg: &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{nil}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := &fakeService{}

			res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{}), directCall{svc: svc}, tt.msg)

			if res.out != nil || res.err != nil {
				t.Errorf("OnUserMessageCallback() = %v, %v, want nil, nil", res.out, res.err)
			}
			if len(svc.saves) != 0 {
				t.Errorf("saves = %d, want 0", len(svc.saves))
			}
			if v, ok := res.state[defaultKey]; ok {
				t.Errorf("state[%s] = %v, want no such key", defaultKey, v)
			}
		})
	}
}

func TestRunner_NoArtifactService_StopsPluginChain(t *testing.T) {
	recorderCalls := 0
	recorder, err := plugin.New(plugin.Config{
		Name: "recorder",
		OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
			recorderCalls++
			return nil, nil
		},
	})
	if err != nil {
		t.Fatalf("plugin.New() error = %v", err)
	}
	a := newTestApp(t, nil, newPlugin(t, savefilesasartifacts.Config{}), recorder)
	a.createSession(t, sessionID, nil)

	a.mustRun(t, sessionID, userMessage(blobPart("test.pdf", []byte("abc"), "application/pdf")))

	if recorderCalls != 0 {
		t.Errorf("recorder OnUserMessageCallback calls = %d, want 0", recorderCalls)
	}
	want := []*genai.Part{{InlineData: &genai.Blob{DisplayName: "test.pdf", Data: []byte("abc"), MIMEType: "application/pdf"}}}
	if diff := cmp.Diff(want, a.userContents[sessionID][0].Parts); diff != "" {
		t.Errorf("UserContent parts mismatch (-want +got):\n%s", diff)
	}
	if v, ok := a.storedState(t, sessionID)[defaultKey]; ok {
		t.Errorf("stored state[%s] = %v, want no such key", defaultKey, v)
	}
}

func TestOnUserMessage_MixedSuccessAndFailureAndInvalidFileName(t *testing.T) {
	t.Run("service error", func(t *testing.T) {
		svc := &fakeService{
			saveFn: func(_ context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
				if req.FileName == "failure.pdf" {
					return nil, errors.New("storage error")
				}
				return &artifact.SaveResponse{Version: 1}, nil
			},
			lookupFn: uriLookup("gs://bucket", "application/octet-stream"),
		}
		failed := blobPart("failure.pdf", []byte("failure data"), "application/pdf")
		msg := userMessage(blobPart("success.pdf", []byte("success data"), "application/pdf"), failed)

		res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{}), directCall{svc: svc}, msg)

		if res.err != nil {
			t.Fatalf("OnUserMessageCallback() error = %v", res.err)
		}
		if res.out == nil {
			t.Fatal("OnUserMessageCallback() content = nil, want the replacement")
		}
		want := []*genai.Part{
			{Text: `[Uploaded Artifact: "success.pdf"]`},
			{FileData: &genai.FileData{FileURI: "gs://bucket/success.pdf/versions/1", MIMEType: "application/pdf", DisplayName: "success.pdf"}},
			{InlineData: &genai.Blob{DisplayName: "failure.pdf", Data: []byte("failure data"), MIMEType: "application/pdf"}},
		}
		if diff := cmp.Diff(want, res.out.Parts); diff != "" {
			t.Errorf("parts mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(map[string]any{"success.pdf": int64(1)}, res.state[defaultKey]); diff != "" {
			t.Errorf("pending delta mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("only blob fails to save", func(t *testing.T) {
		svc := &fakeService{
			saveFn: func(context.Context, *artifact.SaveRequest) (*artifact.SaveResponse, error) {
				return nil, errors.New("storage error")
			},
		}
		msg := userMessage(blobPart("failure.pdf", []byte("failure data"), "application/pdf"))

		res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{}), directCall{svc: svc}, msg)

		if res.err != nil {
			t.Fatalf("OnUserMessageCallback() error = %v", res.err)
		}
		if diff := cmp.Diff([]string{"failure.pdf"}, svc.savedNames()); diff != "" {
			t.Errorf("saved names mismatch (-want +got):\n%s", diff)
		}
		if res.out != nil {
			t.Errorf("OnUserMessageCallback() content = %v, want nil", res.out)
		}
		if v, ok := res.state[defaultKey]; ok {
			t.Errorf("state[%s] = %v, want no such key", defaultKey, v)
		}
	})

	t.Run("file name with a path separator", func(t *testing.T) {
		msg := userMessage(blobPart("bad/name.pdf", []byte("bad"), "application/pdf"), blobPart("good.pdf", []byte("good"), "application/pdf"))

		res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{DisableFileReference: true}), directCall{svc: artifact.InMemoryService()}, msg)

		if res.err != nil {
			t.Fatalf("OnUserMessageCallback() error = %v", res.err)
		}
		if res.out == nil {
			t.Fatal("OnUserMessageCallback() content = nil, want the replacement")
		}
		want := []*genai.Part{
			{InlineData: &genai.Blob{DisplayName: "bad/name.pdf", Data: []byte("bad"), MIMEType: "application/pdf"}},
			{Text: `[Uploaded Artifact: "good.pdf"]`},
		}
		if diff := cmp.Diff(want, res.out.Parts); diff != "" {
			t.Errorf("parts mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(map[string]any{"good.pdf": int64(1)}, res.state[defaultKey]); diff != "" {
			t.Errorf("pending delta mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestOnUserMessage_CancelledContextPropagatesSaveError(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	svc := &fakeService{
		saveFn: func(ctx context.Context, _ *artifact.SaveRequest) (*artifact.SaveResponse, error) {
			cancel()
			return nil, fmt.Errorf("upload aborted: %w", ctx.Err())
		},
	}
	msg := userMessage(blobPart("secret.pdf", []byte("abc"), "application/pdf"))

	res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{}), directCall{svc: svc, ctx: ctx}, msg)

	if !errors.Is(res.err, context.Canceled) {
		t.Fatalf("OnUserMessageCallback() error = %v, want an error wrapping context.Canceled", res.err)
	}
	if res.out != nil {
		t.Errorf("OnUserMessageCallback() content = %v, want nil", res.out)
	}
	if !strings.Contains(res.err.Error(), "savefilesasartifacts: saving part 0") {
		t.Errorf("error %q does not name the part", res.err)
	}
	if strings.Contains(res.err.Error(), "secret.pdf") {
		t.Errorf("error %q contains the file name", res.err)
	}
}

func TestOnUserMessage_FileSizeBoundaryAndMixedSizes(t *testing.T) {
	svc := &fakeService{lookupFn: uriLookup("gs://bucket", "application/octet-stream")}
	msg := userMessage(
		blobPart("exact_20mb.bin", make([]byte, 20*1024*1024), ""),
		blobPart("over_by_1.bin", make([]byte, 20*1024*1024+1), ""),
		blobPart("twenty_five.bin", make([]byte, 25*1024*1024), ""),
	)

	res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{}), directCall{svc: svc}, msg)

	if res.err != nil {
		t.Fatalf("OnUserMessageCallback() error = %v", res.err)
	}
	if diff := cmp.Diff([]string{"exact_20mb.bin"}, svc.savedNames()); diff != "" {
		t.Errorf("saved names mismatch (-want +got):\n%s", diff)
	}
	if res.out == nil {
		t.Fatal("OnUserMessageCallback() content = nil, want the replacement")
	}
	want := []*genai.Part{
		{Text: `[Uploaded Artifact: "exact_20mb.bin"]`},
		{FileData: &genai.FileData{FileURI: "gs://bucket/exact_20mb.bin/versions/1", MIMEType: "application/octet-stream", DisplayName: "exact_20mb.bin"}},
		{Text: "[Upload Error: File over_by_1.bin (20.00 MB) exceeds the maximum supported size of 20MB. Please upload a smaller file.]"},
		{Text: "[Upload Error: File twenty_five.bin (25.00 MB) exceeds the maximum supported size of 20MB. Please upload a smaller file.]"},
	}
	// A diff that printed a kept 20 MiB blob would be unreadable.
	summarize := cmp.Transformer("blobSize", func(b *genai.Blob) string {
		if b == nil {
			return "<nil>"
		}
		return fmt.Sprintf("%s: %d bytes", b.DisplayName, len(b.Data))
	})
	if diff := cmp.Diff(want, res.out.Parts, summarize); diff != "" {
		t.Errorf("parts mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"exact_20mb.bin": int64(1)}, res.state[defaultKey]); diff != "" {
		t.Errorf("pending delta mismatch (-want +got):\n%s", diff)
	}
}

func TestOnUserMessage_OversizedBlobWithoutDisplayNameNamesGeneratedFile(t *testing.T) {
	svc := &fakeService{}
	msg := userMessage(
		genai.NewPartFromText("intro"),
		blobPart("", make([]byte, 20*1024*1024+1), "application/pdf"),
	)

	res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{}), directCall{svc: svc, invocationID: "inv_1"}, msg)

	if res.err != nil {
		t.Fatalf("OnUserMessageCallback() error = %v", res.err)
	}
	if len(svc.saves) != 0 {
		t.Errorf("saves = %d, want 0", len(svc.saves))
	}
	want := userMessage(
		&genai.Part{Text: "intro"},
		&genai.Part{Text: "[Upload Error: File artifact_inv_1_1 (20.00 MB) exceeds the maximum supported size of 20MB. Please upload a smaller file.]"},
	)
	if diff := cmp.Diff(want, res.out); diff != "" {
		t.Errorf("OnUserMessageCallback() content mismatch (-want +got):\n%s", diff)
	}
}

func TestPendingDelta_EdgeCasesAndStateTypes(t *testing.T) {
	t.Run("on-user-message merges into the pending delta", func(t *testing.T) {
		p := newPlugin(t, savefilesasartifacts.Config{DisableFileReference: true})
		oversized := userMessage(blobPart("big.bin", make([]byte, 20*1024*1024+1), ""))

		res := callOnUserMessage(t, p, directCall{svc: &fakeService{}}, oversized)

		if res.err != nil {
			t.Fatalf("OnUserMessageCallback(oversized) error = %v", res.err)
		}
		if diff := cmp.Diff(map[string]any{}, res.state[defaultKey]); diff != "" {
			t.Errorf("pending delta after an oversized-only message mismatch (-want +got):\n%s", diff)
		}

		svc := &fakeService{saveFn: func(context.Context, *artifact.SaveRequest) (*artifact.SaveResponse, error) {
			return &artifact.SaveResponse{Version: 5}, nil
		}}
		prior := map[string]any{defaultKey: map[string]any{"prior.txt": int64(2)}}

		res = callOnUserMessage(t, p, directCall{svc: svc, state: prior}, userMessage(blobPart("new.txt", []byte("new"), "text/plain")))

		if res.err != nil {
			t.Fatalf("OnUserMessageCallback(new.txt) error = %v", res.err)
		}
		if diff := cmp.Diff(map[string]any{"prior.txt": int64(2), "new.txt": int64(5)}, res.state[defaultKey]); diff != "" {
			t.Errorf("merged pending delta mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("before-agent reports integral numbers only", func(t *testing.T) {
		a := newTestApp(t, &fakeService{}, newPlugin(t, savefilesasartifacts.Config{}))
		a.createSession(t, sessionID, map[string]any{defaultKey: map[string]any{
			"int64_file": int64(1),
			"int_file":   2,
			"float_file": float64(3),
			"frac_float": 3.5,
			"bad_type":   "not-a-version",
		}})

		events := a.mustRun(t, sessionID, genai.NewContentFromText("hi", genai.RoleUser))

		if diff := cmp.Diff(map[string]int64{"int64_file": 1, "int_file": 2, "float_file": 3}, events[0].Actions.ArtifactDelta); diff != "" {
			t.Errorf("first event ArtifactDelta mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(map[string]any{defaultKey: map[string]any{}}, events[0].Actions.StateDelta); diff != "" {
			t.Errorf("first event StateDelta mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("before-agent clears a value that is not a map", func(t *testing.T) {
		a := newTestApp(t, &fakeService{}, newPlugin(t, savefilesasartifacts.Config{}))
		a.createSession(t, sessionID, map[string]any{defaultKey: "corrupt"})

		events := a.mustRun(t, sessionID, genai.NewContentFromText("hi", genai.RoleUser))

		if diff := cmp.Diff(map[string]int64{}, events[0].Actions.ArtifactDelta, cmpopts.EquateEmpty()); diff != "" {
			t.Errorf("first event ArtifactDelta mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(map[string]any{defaultKey: map[string]any{}}, events[0].Actions.StateDelta); diff != "" {
			t.Errorf("first event StateDelta mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("before-agent leaves an empty pending delta alone", func(t *testing.T) {
		a := newTestApp(t, &fakeService{}, newPlugin(t, savefilesasartifacts.Config{}))
		a.createSession(t, sessionID, map[string]any{defaultKey: map[string]any{}})

		events := a.mustRun(t, sessionID, genai.NewContentFromText("hi", genai.RoleUser))

		// Only the agent's reply: no event carries a state or artifact delta.
		if len(events) != 1 || events[0].Content == nil {
			t.Fatalf("Run() yielded %d events, want only the reply", len(events))
		}
		if len(events[0].Actions.StateDelta) != 0 || len(events[0].Actions.ArtifactDelta) != 0 {
			t.Errorf("reply event actions = %+v, want no deltas", events[0].Actions)
		}
	})

	t.Run("on-user-message fails on a value that is not a map", func(t *testing.T) {
		state := map[string]any{defaultKey: "corrupt"}

		res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{DisableFileReference: true}), directCall{svc: &fakeService{}, state: state},
			userMessage(blobPart("a.txt", []byte("a"), "text/plain")))

		if res.err == nil || !strings.Contains(res.err.Error(), "holds string, want map[string]any") {
			t.Errorf("OnUserMessageCallback() error = %v, want one containing %q", res.err, "holds string, want map[string]any")
		}
		if res.out != nil {
			t.Errorf("OnUserMessageCallback() content = %v, want nil", res.out)
		}
	})
}

func TestOnUserMessage_ImmutabilityDuplicateNamesAndUserScope(t *testing.T) {
	svc := artifact.InMemoryService()
	part1 := blobPart("user:shared.txt", []byte("v1"), "text/plain")
	part2 := blobPart("user:shared.txt", []byte("v2"), "text/plain")
	msg := &genai.Content{Role: "custom_role", Parts: []*genai.Part{part1, part2}}

	res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{}), directCall{svc: svc}, msg)

	if res.err != nil {
		t.Fatalf("OnUserMessageCallback() error = %v", res.err)
	}
	if len(msg.Parts) != 2 || msg.Parts[0] != part1 || msg.Parts[1] != part2 || part1.InlineData == nil || !bytes.Equal(part1.InlineData.Data, []byte("v1")) {
		t.Errorf("the caller's message changed: %+v", msg.Parts)
	}
	want := &genai.Content{Role: "custom_role", Parts: []*genai.Part{
		{Text: `[Uploaded Artifact: "user:shared.txt"]`},
		{Text: `[Uploaded Artifact: "user:shared.txt"]`},
	}}
	if diff := cmp.Diff(want, res.out); diff != "" {
		t.Errorf("OnUserMessageCallback() content mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"user:shared.txt": int64(2)}, res.state[defaultKey]); diff != "" {
		t.Errorf("pending delta mismatch (-want +got):\n%s", diff)
	}

	// Another session of the same user sees a "user:" artifact.
	v1, err := svc.Load(t.Context(), &artifact.LoadRequest{AppName: appName, UserID: userID, SessionID: "other", FileName: "user:shared.txt", Version: 1})
	if err != nil {
		t.Fatalf("Load(version 1) error = %v", err)
	}
	if diff := cmp.Diff(blobPart("user:shared.txt", []byte("v1"), "text/plain"), v1.Part); diff != "" {
		t.Errorf("Load(version 1) mismatch (-want +got):\n%s", diff)
	}
	v2, err := svc.Load(t.Context(), &artifact.LoadRequest{AppName: appName, UserID: userID, SessionID: "other", FileName: "user:shared.txt", Version: 2})
	if err != nil {
		t.Fatalf("Load(version 2) error = %v", err)
	}
	if diff := cmp.Diff(blobPart("user:shared.txt", []byte("v2"), "text/plain"), v2.Part); diff != "" {
		t.Errorf("Load(version 2) mismatch (-want +got):\n%s", diff)
	}
}

func TestPlugin_ConcurrentInvocations(t *testing.T) {
	const n = 10
	a := newTestApp(t, artifact.InMemoryService(), newPlugin(t, savefilesasartifacts.Config{}))
	for i := range n {
		a.createSession(t, fmt.Sprintf("s%d", i), nil)
	}
	events := make([][]*session.Event, n)
	errs := make([]error, n)

	var wg sync.WaitGroup
	for i := range n {
		wg.Go(func() {
			msg := userMessage(blobPart(fmt.Sprintf("file_%d.txt", i), []byte("data"), "text/plain"))
			events[i], errs[i] = a.run(t.Context(), fmt.Sprintf("s%d", i), msg)
		})
	}
	wg.Wait()

	for i := range n {
		if errs[i] != nil || len(events[i]) == 0 {
			t.Errorf("session s%d: Run() = %d events, error %v", i, len(events[i]), errs[i])
			continue
		}
		name := fmt.Sprintf("file_%d.txt", i)
		if diff := cmp.Diff(map[string]int64{name: 1}, events[i][0].Actions.ArtifactDelta); diff != "" {
			t.Errorf("session s%d first event ArtifactDelta mismatch (-want +got):\n%s", i, diff)
		}
		if diff := cmp.Diff(map[string]any{}, a.storedState(t, fmt.Sprintf("s%d", i))[defaultKey]); diff != "" {
			t.Errorf("session s%d stored pending delta mismatch (-want +got):\n%s", i, diff)
		}
	}
}

func TestOnUserMessage_SavedCopyIsIndependentOfMessage(t *testing.T) {
	svc := &fakeService{}
	msg := userMessage(blobPart("a.pdf", []byte("abc"), "application/pdf"))

	res := callOnUserMessage(t, newPlugin(t, savefilesasartifacts.Config{DisableFileReference: true}), directCall{svc: svc}, msg)
	msg.Parts[0].InlineData.DisplayName = "mutated"
	msg.Parts[0].InlineData.Data[0] = 'X'

	if res.err != nil {
		t.Fatalf("OnUserMessageCallback() error = %v", res.err)
	}
	if len(svc.saves) != 1 {
		t.Fatalf("saves = %d, want 1", len(svc.saves))
	}
	want := &genai.Part{InlineData: &genai.Blob{DisplayName: "a.pdf", Data: []byte("abc"), MIMEType: "application/pdf"}}
	if diff := cmp.Diff(want, svc.saves[0].Part); diff != "" {
		t.Errorf("kept part mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(userMessage(&genai.Part{Text: `[Uploaded Artifact: "a.pdf"]`}), res.out); diff != "" {
		t.Errorf("OnUserMessageCallback() content mismatch (-want +got):\n%s", diff)
	}
}
