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

package artifact_test

import (
	"context"
	"errors"
	"io/fs"
	"slices"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	artifactinternal "google.golang.org/adk/v2/internal/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

func forwardingParent(t *testing.T) (agent.Artifacts, *session.EventActions) {
	t.Helper()
	actions := &session.EventActions{}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Artifacts: &artifactinternal.Artifacts{
			Service: artifact.InMemoryService(), AppName: "parent_app", UserID: "parent_user", SessionID: "parent_session",
		},
	})
	return agent.NewToolContext(ctx, "call", actions, nil).Artifacts(), actions
}

func TestForwardingService(t *testing.T) {
	parent, actions := forwardingParent(t)
	service := artifactinternal.NewForwardingService(parent)
	ctx := t.Context()
	for _, text := range []string{"first", "second"} {
		_, err := service.Save(ctx, &artifact.SaveRequest{
			AppName: "child_app", UserID: "child_user", SessionID: "child_session",
			FileName: "report.txt", Part: genai.NewPartFromText(text),
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if actions.ArtifactDelta["report.txt"] != 2 {
		t.Error("parent delta does not contain the latest saved version")
	}
	if got, err := parent.Load(ctx, "report.txt"); err != nil || got.Part.Text != "second" {
		t.Fatal("forwarded save did not reach the parent")
	}
	for _, tc := range []struct {
		version int64
		text    string
	}{{0, "second"}, {1, "first"}} {
		got, err := service.Load(ctx, &artifact.LoadRequest{
			AppName: "child_app", UserID: "child_user", SessionID: "child_session", FileName: "report.txt", Version: tc.version,
		})
		if err != nil || got.Part.Text != tc.text {
			t.Error("load did not return the requested parent version")
		}
	}
	listed, err := service.List(ctx, &artifact.ListRequest{AppName: "child_app", UserID: "child_user", SessionID: "child_session"})
	if err != nil || !slices.Equal(listed.FileNames, []string{"report.txt"}) {
		t.Fatal("list did not use parent scope")
	}

	versionsReq := &artifact.VersionsRequest{AppName: "child_app", UserID: "child_user", SessionID: "child_session", FileName: "report.txt"}
	versionsBefore := *versionsReq
	versions, err := service.Versions(ctx, versionsReq)
	if err != nil || len(versions.Versions) != 2 || !slices.Contains(versions.Versions, int64(1)) || !slices.Contains(versions.Versions, int64(2)) {
		t.Error("versions did not use parent scope")
	}
	if *versionsReq != versionsBefore {
		t.Error("versions request was mutated")
	}

	metadataReq := &artifact.GetArtifactVersionRequest{AppName: "child_app", UserID: "child_user", SessionID: "child_session", FileName: "report.txt", Version: 1}
	metadataBefore := *metadataReq
	metadata, err := service.GetArtifactVersion(ctx, metadataReq)
	if err != nil || metadata.ArtifactVersion.Version != 1 {
		t.Error("metadata did not use parent scope and requested version")
	}
	if *metadataReq != metadataBefore {
		t.Error("metadata request was mutated")
	}

	deleteReq := &artifact.DeleteRequest{AppName: "child_app", UserID: "child_user", SessionID: "child_session", FileName: "report.txt", Version: 2}
	deleteBefore := *deleteReq
	if err := service.Delete(ctx, deleteReq); err != nil {
		t.Fatal(err)
	}
	if *deleteReq != deleteBefore {
		t.Error("delete request was mutated")
	}
	if _, err := parent.LoadVersion(ctx, "report.txt", 2); !errors.Is(err, fs.ErrNotExist) {
		t.Error("delete did not remove the requested parent version")
	}
	if _, err := parent.LoadVersion(ctx, "report.txt", 1); err != nil {
		t.Error("delete removed another version")
	}
}

func TestForwardingService_ParallelSaves(t *testing.T) {
	parent, actions := forwardingParent(t)
	service := artifactinternal.NewForwardingService(parent)
	const saves = 64
	var wg sync.WaitGroup
	start := make(chan struct{})
	for range saves {
		wg.Go(func() {
			<-start
			_, err := service.Save(t.Context(), &artifact.SaveRequest{FileName: "report.txt", Part: genai.NewPartFromText("result")})
			if err != nil {
				t.Error("parallel save failed")
			}
		})
	}
	close(start)
	wg.Wait()
	if actions.ArtifactDelta["report.txt"] != saves {
		t.Error("parallel saves lost the latest artifact version")
	}
}

type failingParent struct {
	agent.Artifacts
	err     error
	loadErr error
}

func (p failingParent) Save(context.Context, string, *genai.Part) (*artifact.SaveResponse, error) {
	return nil, p.err
}

func (p failingParent) Load(context.Context, string) (*artifact.LoadResponse, error) {
	return nil, p.loadErr
}

func (p failingParent) LoadVersion(context.Context, string, int) (*artifact.LoadResponse, error) {
	return nil, p.err
}
func (p failingParent) List(context.Context) (*artifact.ListResponse, error) { return nil, p.err }

func TestForwardingService_Errors(t *testing.T) {
	backendErr := errors.New("backend failed")
	loadErr := errors.New("latest load failed")
	for _, tc := range []struct {
		name   string
		parent agent.Artifacts
	}{
		{name: "no parent service"},
		{name: "custom parent", parent: failingParent{err: backendErr, loadErr: loadErr}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			service := artifactinternal.NewForwardingService(tc.parent)
			for _, operation := range []struct {
				name string
				want error
				run  func() error
			}{
				{name: "save", want: backendErr, run: func() error { _, err := service.Save(t.Context(), &artifact.SaveRequest{}); return err }},
				{name: "load latest", want: loadErr, run: func() error { _, err := service.Load(t.Context(), &artifact.LoadRequest{}); return err }},
				{name: "load version", want: backendErr, run: func() error { _, err := service.Load(t.Context(), &artifact.LoadRequest{Version: 1}); return err }},
				{name: "list", want: backendErr, run: func() error { _, err := service.List(t.Context(), &artifact.ListRequest{}); return err }},
				{name: "delete", want: errors.ErrUnsupported, run: func() error { return service.Delete(t.Context(), &artifact.DeleteRequest{}) }},
				{name: "versions", want: errors.ErrUnsupported, run: func() error { _, err := service.Versions(t.Context(), &artifact.VersionsRequest{}); return err }},
				{name: "metadata", want: errors.ErrUnsupported, run: func() error {
					_, err := service.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{})
					return err
				}},
			} {
				t.Run(operation.name, func(t *testing.T) {
					err := operation.run()
					if err == nil {
						t.Fatal("operation unexpectedly succeeded")
					}
					if tc.parent != nil {
						if !errors.Is(err, operation.want) {
							t.Error("operation did not preserve the expected error")
						}
					} else if errors.Is(err, errors.ErrUnsupported) {
						t.Error("missing service was treated as an unsupported operation")
					}
				})
			}
		})
	}
}
