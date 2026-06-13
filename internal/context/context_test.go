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

package context

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/uuid"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/platform"
)

type fakeArtifacts struct {
	version     int64
	nextVersion func() int64
}

func (f *fakeArtifacts) Save(ctx context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	if f.nextVersion != nil {
		return &artifact.SaveResponse{Version: f.nextVersion()}, nil
	}
	return &artifact.SaveResponse{Version: f.version}, nil
}

func (f *fakeArtifacts) List(ctx context.Context) (*artifact.ListResponse, error) {
	return nil, nil
}

func (f *fakeArtifacts) Load(ctx context.Context, name string) (*artifact.LoadResponse, error) {
	return nil, nil
}

func (f *fakeArtifacts) LoadVersion(ctx context.Context, name string, version int) (*artifact.LoadResponse, error) {
	return nil, nil
}

func TestReadonlyContext(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{})
	readonly := NewReadonlyContext(inv)

	if got, ok := readonly.(agent.InvocationContext); ok {
		t.Errorf("ReadonlyContext(%+T) is unexpectedly an InvocationContext", got)
	}
}

func TestCallbackContext(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{})
	callback := NewCallbackContext(inv)

	if _, ok := callback.(agent.ReadonlyContext); !ok {
		t.Errorf("CallbackContext(%+T) is unexpectedly not a ReadonlyContext", callback)
	}
	if got, ok := callback.(agent.InvocationContext); ok {
		t.Errorf("CallbackContext(%+T) is unexpectedly an InvocationContext", got)
	}
}

func TestCallbackContextArtifactsSaveKeepsNewestVersion(t *testing.T) {
	t.Parallel()

	artifactDelta := make(map[string]int64)
	fake := &fakeArtifacts{version: 1}
	inv := NewInvocationContext(t.Context(), InvocationContextParams{Artifacts: fake})
	callbackCtx := NewCallbackContextWithDelta(inv, nil, artifactDelta)

	ctx := context.Background()
	if _, err := callbackCtx.Artifacts().Save(ctx, "file.txt", nil); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}
	if got := artifactDelta["file.txt"]; got != 1 {
		t.Fatalf("expected version 1 after first save, got %d", got)
	}

	fake.version = 3
	if _, err := callbackCtx.Artifacts().Save(ctx, "file.txt", nil); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}
	if got := artifactDelta["file.txt"]; got != 3 {
		t.Fatalf("expected version 3 after newer save, got %d", got)
	}

	fake.version = 2
	if _, err := callbackCtx.Artifacts().Save(ctx, "file.txt", nil); err != nil {
		t.Fatalf("Save returned error: %v", err)
	}
	if got := artifactDelta["file.txt"]; got != 3 {
		t.Fatalf("expected version 3 after older save, got %d", got)
	}
}

func TestCallbackContextArtifactsSaveConcurrentKeepsNewestVersion(t *testing.T) {
	t.Parallel()

	artifactDelta := make(map[string]int64)
	var version atomic.Int64
	fake := &fakeArtifacts{nextVersion: func() int64 {
		return version.Add(1)
	}}
	inv := NewInvocationContext(t.Context(), InvocationContextParams{Artifacts: fake})
	callbackCtx := NewCallbackContextWithDelta(inv, nil, artifactDelta)

	const saves = 100
	var wg sync.WaitGroup
	errs := make(chan error, saves)
	for range saves {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := callbackCtx.Artifacts().Save(context.Background(), "file.txt", nil); err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Fatalf("Save returned error: %v", err)
	}
	if got := artifactDelta["file.txt"]; got != saves {
		t.Fatalf("expected version %d after concurrent saves, got %d", saves, got)
	}
}

type testKey struct{}

func TestWithContext(t *testing.T) {
	baseCtx := t.Context()
	inv := NewInvocationContext(baseCtx, InvocationContextParams{
		Branch: "test-branch",
	})

	key := testKey{}
	val := "val"
	got := inv.WithContext(context.WithValue(baseCtx, key, val))

	if got.Value(key) != val {
		t.Errorf("WithContext() did not update context")
	}
	if diff := cmp.Diff(inv, got, cmp.AllowUnexported(InvocationContext{}), cmpopts.IgnoreFields(InvocationContext{}, "Context")); diff != "" {
		t.Errorf("WithContext() mismatch (-want +got):\n%s", diff)
	}
}

func TestNewInvocationContextGeneratesIDWithProvider(t *testing.T) {
	ctx := platform.WithUUIDProvider(t.Context(), func() string { return "fixed" })
	inv := NewInvocationContext(ctx, InvocationContextParams{})

	if got, want := inv.InvocationID(), "e-fixed"; got != want {
		t.Errorf("InvocationID() = %q, want %q", got, want)
	}
}

func TestNewInvocationContextRespectsExplicitID(t *testing.T) {
	// An explicit InvocationID must be used verbatim, leaving the provider unused.
	ctx := platform.WithUUIDProvider(t.Context(), func() string { return "fixed" })
	inv := NewInvocationContext(ctx, InvocationContextParams{InvocationID: "explicit"})

	if got, want := inv.InvocationID(), "explicit"; got != want {
		t.Errorf("InvocationID() = %q, want %q", got, want)
	}
}

func TestNewInvocationContextDefaultID(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{})

	id := inv.InvocationID()
	rest, ok := strings.CutPrefix(id, "e-")
	if !ok {
		t.Fatalf("InvocationID() = %q, want \"e-\" prefix", id)
	}
	if _, err := uuid.Parse(rest); err != nil {
		t.Errorf("InvocationID() = %q, suffix not a valid UUID: %v", id, err)
	}
}

func TestInvocationContext_LiveSessionResumptionHandle(t *testing.T) {
	inv := NewInvocationContext(t.Context(), InvocationContextParams{})

	iCtx, ok := inv.(*InvocationContext)
	if !ok {
		t.Fatalf("NewInvocationContext did not return *InvocationContext")
	}

	if iCtx.LiveSessionResumptionHandle() != "" {
		t.Errorf("expected empty handle, got %q", iCtx.LiveSessionResumptionHandle())
	}

	iCtx.SetLiveSessionResumptionHandle("test-handle")
	if iCtx.LiveSessionResumptionHandle() != "test-handle" {
		t.Errorf("expected test-handle, got %q", iCtx.LiveSessionResumptionHandle())
	}
}
