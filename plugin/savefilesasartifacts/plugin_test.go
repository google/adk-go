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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/artifact"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/savefilesasartifacts"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
)

const (
	appName      = "test_app"
	userID       = "test_user"
	sessionID    = "test_session"
	invocationID = "test_invocation_123"
	mb           = 1024 * 1024
)

// gcsLikeService wraps the in-memory service, reporting gs:// canonical URIs
// and failing the saves listed in failOn. versionErr and nilVersion make
// GetArtifactVersion fail or return no version.
type gcsLikeService struct {
	artifact.Service
	failOn     map[string]bool
	versionErr bool
	nilVersion bool
}

func (s *gcsLikeService) Save(ctx context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	if s.failOn[req.FileName] {
		return nil, errors.New("storage error")
	}
	return s.Service.Save(ctx, req)
}

func (s *gcsLikeService) GetArtifactVersion(ctx context.Context, req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	if s.nilVersion {
		return &artifact.GetArtifactVersionResponse{}, nil
	}
	resp := &artifact.GetArtifactVersionResponse{ArtifactVersion: &artifact.ArtifactVersion{
		Version:      req.Version,
		CanonicalURI: fmt.Sprintf("gs://mock-bucket/%s/versions/%d", req.FileName, req.Version),
		MimeType:     "application/pdf",
	}}
	if s.versionErr {
		// A response alongside the error must still be ignored.
		return resp, errors.New("lookup error")
	}
	return resp, nil
}

func newSession(t *testing.T) session.Session {
	t.Helper()
	resp, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{
		AppName: appName, UserID: userID, SessionID: sessionID,
	})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	return resp.Session
}

func newInvocationContext(t *testing.T, sess session.Session, svc artifact.Service) agent.InvocationContext {
	t.Helper()
	var artifacts agent.Artifacts
	if svc != nil {
		artifacts = &artifactinternal.Artifacts{Service: svc, AppName: appName, UserID: userID, SessionID: sessionID}
	}
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Artifacts:    artifacts,
		Session:      sess,
		InvocationID: invocationID,
	})
}

func blobPart(name, mimeType string, data []byte) *genai.Part {
	return &genai.Part{InlineData: &genai.Blob{DisplayName: name, MIMEType: mimeType, Data: data}}
}

func placeholder(name string) *genai.Part {
	return genai.NewPartFromText(`[Uploaded Artifact: "` + name + `"]`)
}

func gsRef(name, mimeType string, version int64) *genai.Part {
	return &genai.Part{FileData: &genai.FileData{
		FileURI:     fmt.Sprintf("gs://mock-bucket/%s/versions/%d", name, version),
		MIMEType:    mimeType,
		DisplayName: name,
	}}
}

func uploadError(name, size string) *genai.Part {
	return genai.NewPartFromText("[Upload Error: File " + name + " (" + size +
		" MB) exceeds the maximum supported size of 20MB. Please upload a smaller file.]")
}

func TestOnUserMessage(t *testing.T) {
	failurePart := blobPart("failure.pdf", "application/pdf", []byte("failure data"))
	const docxMIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

	tests := []struct {
		name string
		cfg  savefilesasartifacts.Config
		// inMemoryOnly uses the plain in-memory service (memory:// URIs) as
		// the config's ArtifactService instead of the gs:// wrapper.
		inMemoryOnly bool
		// noConfigService leaves Config.ArtifactService nil.
		noConfigService bool
		// noArtifacts leaves the invocation without an artifact service.
		noArtifacts bool
		failOn      map[string]bool
		versionErr  bool
		nilVersion  bool
		parts       []*genai.Part
		// wantParts nil means the callback returns nil content.
		wantParts []*genai.Part
		wantDelta map[string]int64
	}{
		{
			name:      "display name",
			parts:     []*genai.Part{blobPart("test_document.pdf", "application/pdf", []byte("test data"))},
			wantParts: []*genai.Part{placeholder("test_document.pdf"), gsRef("test_document.pdf", "application/pdf", 1)},
			wantDelta: map[string]int64{"test_document.pdf": 1},
		},
		{
			name:      "skip file reference",
			cfg:       savefilesasartifacts.Config{SkipFileReference: true},
			parts:     []*genai.Part{blobPart("test_document.pdf", "application/pdf", []byte("test data"))},
			wantParts: []*genai.Part{placeholder("test_document.pdf")},
			wantDelta: map[string]int64{"test_document.pdf": 1},
		},
		{
			name:  "no display name",
			parts: []*genai.Part{blobPart("", "application/pdf", []byte("test data"))},
			wantParts: []*genai.Part{
				placeholder("artifact_test_invocation_123_0"),
				gsRef("artifact_test_invocation_123_0", "application/pdf", 1),
			},
			wantDelta: map[string]int64{"artifact_test_invocation_123_0": 1},
		},
		{
			name: "several files without display name",
			parts: []*genai.Part{
				blobPart("", "text/plain", []byte("first")),
				genai.NewPartFromText("between"),
				blobPart("", "text/plain", []byte("second")),
			},
			wantParts: []*genai.Part{
				placeholder("artifact_test_invocation_123_0"),
				gsRef("artifact_test_invocation_123_0", "text/plain", 1),
				genai.NewPartFromText("between"),
				placeholder("artifact_test_invocation_123_2"),
				gsRef("artifact_test_invocation_123_2", "text/plain", 1),
			},
			wantDelta: map[string]int64{"artifact_test_invocation_123_0": 1, "artifact_test_invocation_123_2": 1},
		},
		{
			name:      "percent sign in file name keeps file reference",
			parts:     []*genai.Part{blobPart("50% off.pdf", "application/pdf", []byte("test data"))},
			wantParts: []*genai.Part{placeholder("50% off.pdf"), gsRef("50% off.pdf", "application/pdf", 1)},
			wantDelta: map[string]int64{"50% off.pdf": 1},
		},
		{
			name:       "version lookup error skips file reference",
			versionErr: true,
			parts:      []*genai.Part{blobPart("test.pdf", "application/pdf", []byte("test data"))},
			wantParts:  []*genai.Part{placeholder("test.pdf")},
			wantDelta:  map[string]int64{"test.pdf": 1},
		},
		{
			name:       "missing version skips file reference",
			nilVersion: true,
			parts:      []*genai.Part{blobPart("test.pdf", "application/pdf", []byte("test data"))},
			wantParts:  []*genai.Part{placeholder("test.pdf")},
			wantDelta:  map[string]int64{"test.pdf": 1},
		},
		{
			name: "multiple files in message",
			parts: []*genai.Part{
				blobPart("file1.txt", "text/plain", []byte("file1 content")),
				genai.NewPartFromText("Some text between files"),
				blobPart("file2.jpg", "image/jpeg", []byte("file2 content")),
			},
			wantParts: []*genai.Part{
				placeholder("file1.txt"),
				gsRef("file1.txt", "text/plain", 1),
				genai.NewPartFromText("Some text between files"),
				placeholder("file2.jpg"),
				gsRef("file2.jpg", "image/jpeg", 1),
			},
			wantDelta: map[string]int64{"file1.txt": 1, "file2.jpg": 1},
		},
		{
			name:        "no artifact service",
			noArtifacts: true,
			parts:       []*genai.Part{blobPart("test.pdf", "application/pdf", []byte("test data"))},
		},
		{
			name: "no parts in message",
		},
		{
			name:  "parts without inline data",
			parts: []*genai.Part{genai.NewPartFromText("Hello world"), genai.NewPartFromText("No files here")},
		},
		{
			name:   "save failure",
			failOn: map[string]bool{"test.pdf": true},
			parts:  []*genai.Part{blobPart("test.pdf", "application/pdf", []byte("test data"))},
		},
		{
			name:   "mixed success and failure",
			failOn: map[string]bool{"failure.pdf": true},
			parts: []*genai.Part{
				blobPart("success.pdf", "application/pdf", []byte("success data")),
				failurePart,
			},
			wantParts: []*genai.Part{placeholder("success.pdf"), gsRef("success.pdf", "application/pdf", 1), failurePart},
			wantDelta: map[string]int64{"success.pdf": 1},
		},
		{
			name:      "placeholder text format",
			parts:     []*genai.Part{blobPart("test file with spaces.docx", docxMIME, []byte("document data"))},
			wantParts: []*genai.Part{placeholder("test file with spaces.docx"), gsRef("test file with spaces.docx", docxMIME, 1)},
			wantDelta: map[string]int64{"test file with spaces.docx": 1},
		},
		{
			name:      "file size exceeds limit",
			parts:     []*genai.Part{blobPart("large_file.pdf", "application/pdf", make([]byte, 21*mb))},
			wantParts: []*genai.Part{uploadError("large_file.pdf", "21.00")},
		},
		{
			name:      "file size at limit",
			parts:     []*genai.Part{blobPart("max_size_file.pdf", "application/pdf", make([]byte, 20*mb))},
			wantParts: []*genai.Part{placeholder("max_size_file.pdf"), gsRef("max_size_file.pdf", "application/pdf", 1)},
			wantDelta: map[string]int64{"max_size_file.pdf": 1},
		},
		{
			name:      "file size just over limit",
			parts:     []*genai.Part{blobPart("slightly_too_large.pdf", "application/pdf", make([]byte, 20*mb+1))},
			wantParts: []*genai.Part{uploadError("slightly_too_large.pdf", "20.00")},
		},
		{
			name: "mixed file sizes",
			parts: []*genai.Part{
				blobPart("small.pdf", "application/pdf", make([]byte, 5*mb)),
				blobPart("large.pdf", "application/pdf", make([]byte, 25*mb)),
			},
			wantParts: []*genai.Part{
				placeholder("small.pdf"),
				gsRef("small.pdf", "application/pdf", 1),
				uploadError("large.pdf", "25.00"),
			},
			wantDelta: map[string]int64{"small.pdf": 1},
		},
		{
			name:            "no artifact service in config skips file reference",
			noConfigService: true,
			parts:           []*genai.Part{blobPart("test.pdf", "application/pdf", []byte("test data"))},
			wantParts:       []*genai.Part{placeholder("test.pdf")},
			wantDelta:       map[string]int64{"test.pdf": 1},
		},
		{
			name:      "missing MIME type falls back to the stored version",
			parts:     []*genai.Part{blobPart("test.pdf", "", []byte("test data"))},
			wantParts: []*genai.Part{placeholder("test.pdf"), gsRef("test.pdf", "application/pdf", 1)},
			wantDelta: map[string]int64{"test.pdf": 1},
		},
		{
			name:         "memory URI skips file reference",
			inMemoryOnly: true,
			parts:        []*genai.Part{blobPart("test.pdf", "application/pdf", []byte("test data"))},
			wantParts:    []*genai.Part{placeholder("test.pdf")},
			wantDelta:    map[string]int64{"test.pdf": 1},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			inMemory := artifact.InMemoryService()
			svc := &gcsLikeService{Service: inMemory, failOn: tc.failOn, versionErr: tc.versionErr, nilVersion: tc.nilVersion}

			cfg := tc.cfg
			switch {
			case tc.inMemoryOnly:
				cfg.ArtifactService = inMemory
			case !tc.noConfigService:
				cfg.ArtifactService = svc
			}
			p, err := savefilesasartifacts.New(cfg)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			var ctxSvc artifact.Service = svc
			if tc.noArtifacts {
				ctxSvc = nil
			}
			sess := newSession(t)
			ictx := newInvocationContext(t, sess, ctxSvc)

			msg := &genai.Content{Role: genai.RoleUser, Parts: tc.parts}
			got, err := p.OnUserMessageCallback()(ictx, msg)
			if err != nil {
				t.Fatalf("OnUserMessageCallback() error = %v", err)
			}

			if tc.wantParts == nil {
				if got != nil {
					t.Fatalf("OnUserMessageCallback() = %+v, want nil", got)
				}
			} else {
				if got == nil {
					t.Fatal("OnUserMessageCallback() = nil, want content")
				}
				if got.Role != genai.RoleUser {
					t.Errorf("Role = %q, want %q", got.Role, genai.RoleUser)
				}
				if diff := cmp.Diff(tc.wantParts, got.Parts); diff != "" {
					t.Errorf("Parts mismatch (-want +got):\n%s", diff)
				}
			}

			// The stored artifact holds the original file.
			for name := range tc.wantDelta {
				resp, err := inMemory.Load(t.Context(), &artifact.LoadRequest{
					AppName: appName, UserID: userID, SessionID: sessionID, FileName: name,
				})
				if err != nil {
					t.Fatalf("Load(%q) error = %v", name, err)
				}
				var want []byte
				for i, part := range tc.parts {
					if part.InlineData == nil {
						continue
					}
					partName := part.InlineData.DisplayName
					if partName == "" {
						partName = fmt.Sprintf("artifact_%s_%d", invocationID, i)
					}
					if partName == name {
						want = part.InlineData.Data
					}
				}
				if !bytes.Equal(resp.Part.InlineData.Data, want) {
					t.Errorf("Load(%q) returned different data than uploaded", name)
				}
			}
			list, err := inMemory.List(t.Context(), &artifact.ListRequest{AppName: appName, UserID: userID, SessionID: sessionID})
			if err != nil {
				t.Fatalf("List() error = %v", err)
			}
			if len(list.FileNames) != len(tc.wantDelta) {
				t.Errorf("saved %v, want %d artifacts", list.FileNames, len(tc.wantDelta))
			}

			ev := &session.Event{}
			if _, err := p.OnEventCallback()(ictx, ev); err != nil {
				t.Fatalf("OnEventCallback() error = %v", err)
			}
			if diff := cmp.Diff(tc.wantDelta, ev.Actions.ArtifactDelta); diff != "" {
				t.Errorf("ArtifactDelta mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDefaultName(t *testing.T) {
	tests := []struct {
		name string
		cfg  savefilesasartifacts.Config
		want string
	}{
		{name: "default", want: "save_files_as_artifacts_plugin"},
		{name: "custom", cfg: savefilesasartifacts.Config{Name: "custom"}, want: "custom"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, err := savefilesasartifacts.New(tc.cfg)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			if got := p.Name(); got != tc.want {
				t.Errorf("Name() = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestArtifactDeltaReporting mirrors adk-python's two-turn check that saved
// versions are reported once per turn. adk-go reports them on the first
// non-partial event of the invocation rather than from BeforeAgentCallback.
func TestArtifactDeltaReporting(t *testing.T) {
	svc := artifact.InMemoryService()
	p, err := savefilesasartifacts.New(savefilesasartifacts.Config{})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	sess := newSession(t)
	ictx := newInvocationContext(t, sess, svc)

	turns := []struct {
		file string
		// agentDelta is the ArtifactDelta the agent's own event already has.
		agentDelta map[string]int64
		wantDelta  map[string]int64
	}{
		{file: "blob.pdf", wantDelta: map[string]int64{"blob.pdf": 1}},
		{file: "blob_2.pdf", wantDelta: map[string]int64{"blob_2.pdf": 1}},
		{
			file:       "blob.pdf",
			agentDelta: map[string]int64{"blob.pdf": 5, "other.txt": 1},
			wantDelta:  map[string]int64{"blob.pdf": 5, "other.txt": 1},
		},
		{
			// The plugin saved version 3, newer than the agent's version 1.
			file:       "blob.pdf",
			agentDelta: map[string]int64{"blob.pdf": 1},
			wantDelta:  map[string]int64{"blob.pdf": 3},
		},
	}
	for i, turn := range turns {
		msg := &genai.Content{Parts: []*genai.Part{blobPart(turn.file, "application/pdf", []byte("test data"))}}
		if _, err := p.OnUserMessageCallback()(ictx, msg); err != nil {
			t.Fatalf("turn %d: OnUserMessageCallback() error = %v", i, err)
		}

		partial := &session.Event{}
		partial.Partial = true
		if _, err := p.OnEventCallback()(ictx, partial); err != nil {
			t.Fatalf("turn %d: OnEventCallback() error = %v", i, err)
		}
		if len(partial.Actions.ArtifactDelta) != 0 {
			t.Errorf("turn %d: partial event ArtifactDelta = %v, want empty", i, partial.Actions.ArtifactDelta)
		}

		ev := &session.Event{Actions: session.EventActions{ArtifactDelta: turn.agentDelta}}
		if _, err := p.OnEventCallback()(ictx, ev); err != nil {
			t.Fatalf("turn %d: OnEventCallback() error = %v", i, err)
		}
		if diff := cmp.Diff(turn.wantDelta, ev.Actions.ArtifactDelta); diff != "" {
			t.Errorf("turn %d: ArtifactDelta mismatch (-want +got):\n%s", i, diff)
		}

		next := &session.Event{}
		if _, err := p.OnEventCallback()(ictx, next); err != nil {
			t.Fatalf("turn %d: OnEventCallback() error = %v", i, err)
		}
		if len(next.Actions.ArtifactDelta) != 0 {
			t.Errorf("turn %d: later event ArtifactDelta = %v, want empty", i, next.Actions.ArtifactDelta)
		}
	}
}

// TestPendingDeltaDoesNotOutliveRun checks that versions saved by a run that
// ends before any event, for example because appending the user event fails,
// are not held by the plugin and do not reach a later run, even one that
// reuses the invocation ID as a resumed invocation does.
func TestPendingDeltaDoesNotOutliveRun(t *testing.T) {
	p, err := savefilesasartifacts.New(savefilesasartifacts.Config{})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	svc := artifact.InMemoryService()

	failed := newInvocationContext(t, newSession(t), svc)
	msg := &genai.Content{Parts: []*genai.Part{blobPart("blob.pdf", "application/pdf", []byte("test data"))}}
	if _, err := p.OnUserMessageCallback()(failed, msg); err != nil {
		t.Fatalf("OnUserMessageCallback() error = %v", err)
	}

	next := newInvocationContext(t, newSession(t), svc)
	ev := &session.Event{}
	if _, err := p.OnEventCallback()(next, ev); err != nil {
		t.Fatalf("OnEventCallback() error = %v", err)
	}
	if len(ev.Actions.ArtifactDelta) != 0 {
		t.Errorf("ArtifactDelta = %v from an earlier run, want empty", ev.Actions.ArtifactDelta)
	}
}

// TestRunner checks the plugin end to end: the user event carries the
// placeholder and the agent's event carries the ArtifactDelta. A custom agent
// runs on the legacy run loop and an llmagent on the node runtime.
func TestRunner(t *testing.T) {
	tests := []struct {
		name     string
		newAgent func() (agent.Agent, error)
	}{
		{
			name: "custom agent",
			newAgent: func() (agent.Agent, error) {
				return agent.New(agent.Config{
					Name: "test_agent",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "test_agent"
							ev.Content = genai.NewContentFromText("done", genai.RoleModel)
							yield(ev, nil)
						}
					},
				})
			},
		},
		{
			name: "llmagent",
			newAgent: func() (agent.Agent, error) {
				return llmagent.New(llmagent.Config{
					Name:  "test_agent",
					Model: &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("done", genai.RoleModel)}},
				})
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			testAgent, err := tc.newAgent()
			if err != nil {
				t.Fatalf("new agent error = %v", err)
			}
			testRunner(t, testAgent)
		})
	}
}

func testRunner(t *testing.T, testAgent agent.Agent) {
	t.Helper()
	ctx := t.Context()
	sessionService := session.InMemoryService()
	artifactService := artifact.InMemoryService()
	p, err := savefilesasartifacts.New(savefilesasartifacts.Config{ArtifactService: artifactService})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:         appName,
		Agent:           testAgent,
		SessionService:  sessionService,
		ArtifactService: artifactService,
		PluginConfig:    runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	if _, err := sessionService.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID}); err != nil {
		t.Fatalf("session Create() error = %v", err)
	}

	msg := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{
		genai.NewPartFromText("here is a file"),
		blobPart("report.pdf", "application/pdf", []byte("pdf bytes")),
	}}
	for _, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	}

	resp, err := sessionService.Get(ctx, &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID})
	if err != nil {
		t.Fatalf("session Get() error = %v", err)
	}
	events := resp.Session.Events()
	if events.Len() != 2 {
		t.Fatalf("got %d events, want 2 (user message and agent reply)", events.Len())
	}

	userEvent := events.At(0)
	wantParts := []*genai.Part{genai.NewPartFromText("here is a file"), placeholder("report.pdf")}
	if diff := cmp.Diff(wantParts, userEvent.Content.Parts); diff != "" {
		t.Errorf("user event parts mismatch (-want +got):\n%s", diff)
	}

	agentEvent := events.At(1)
	if agentEvent.Author != "test_agent" {
		t.Errorf("agent event Author = %q, want %q", agentEvent.Author, "test_agent")
	}
	if diff := cmp.Diff(map[string]int64{"report.pdf": 1}, agentEvent.Actions.ArtifactDelta); diff != "" {
		t.Errorf("ArtifactDelta mismatch (-want +got):\n%s", diff)
	}
}
