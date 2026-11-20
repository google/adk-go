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

package gcsartifact

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"cloud.google.com/go/storage"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/artifact/tests"
	"google.golang.org/adk/v2/platform"
)

// newGCSServiceForTesting creates a *gcsService backed by the in-memory fake. It
// stubs out the retry backoff so retry-heavy tests stay fast.
func newGCSServiceForTesting(bucketName string) *gcsService {
	client := newFakeClient()
	return &gcsService{
		bucketName:    bucketName,
		storageClient: client,
		bucket:        client.bucket(bucketName),
		sleep:         func(ctx context.Context, _ time.Duration) error { return ctx.Err() },
	}
}

// newGCSArtifactServiceForTesting creates a gcsService for the specified bucket using a mocked inmemory client
func newGCSArtifactServiceForTesting(bucketName string) (artifact.Service, error) {
	return newGCSServiceForTesting(bucketName), nil
}

func TestGCSArtifactService(t *testing.T) {
	factory := func(t *testing.T) (artifact.Service, error) {
		return newGCSArtifactServiceForTesting("new")
	}
	tests.TestArtifactService(t, "GCS", factory)
}

func TestDeleteMissingVersion(t *testing.T) {
	svc := newGCSServiceForTesting("bucket")
	if err := svc.Delete(t.Context(), &artifact.DeleteRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "file", Version: 1,
	}); !errors.Is(err, storage.ErrObjectNotExist) {
		t.Errorf("Delete(missing version) = %v, want %v", err, storage.ErrObjectNotExist)
	}
}

func TestDeleteVersionErrors(t *testing.T) {
	for _, tc := range []struct {
		name    string
		err     error
		wantErr error
	}{
		{name: "success"},
		{name: "not found", err: storage.ErrObjectNotExist, wantErr: storage.ErrObjectNotExist},
		{name: "wrapped not found", err: fmt.Errorf("delete: %w", storage.ErrObjectNotExist), wantErr: storage.ErrObjectNotExist},
		{name: "permission denied", err: fs.ErrPermission, wantErr: fs.ErrPermission},
		{name: "canceled", err: context.Canceled, wantErr: context.Canceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := newGCSServiceForTesting("bucket")
			if _, err := svc.Save(t.Context(), saveReq()); err != nil {
				t.Fatalf("Save() failed: %v", err)
			}
			svc.bucket.(*fakeBucket).deleteErr = tc.err

			err := svc.Delete(t.Context(), &artifact.DeleteRequest{
				AppName: "app", UserID: "user", SessionID: "session", FileName: "file", Version: 1,
			})
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("Delete() = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestGCSArtifactVersionFields(t *testing.T) {
	firstCreateTime := time.Date(2026, time.August, 31, 12, 34, 56, 0, time.UTC)
	secondCreateTime := firstCreateTime.Add(time.Minute)
	recreateTime := secondCreateTime.Add(time.Minute)

	for _, tc := range []struct {
		name          string
		fileName      string
		saveSessionID string
		getSessionID  string
		uriPrefix     string
	}{
		{
			name:          "session scoped",
			fileName:      "file.txt",
			saveSessionID: "session",
			getSessionID:  "session",
			uriPrefix:     "gs://bucket/app/user/session/file.txt/",
		},
		{
			name:          "user scoped",
			fileName:      "user:file.txt",
			saveSessionID: "save-session",
			getSessionID:  "irrelevant-session",
			uriPrefix:     "gs://bucket/app/user/user/user:file.txt/",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := newGCSServiceForTesting("bucket")
			fb := svc.bucket.(*fakeBucket)
			fb.setNow(func() time.Time { return firstCreateTime })
			metadata := map[string]any{
				"string":  "value",
				"number":  42,
				"enabled": true,
			}
			if _, err := svc.Save(t.Context(), &artifact.SaveRequest{
				AppName: "app", UserID: "user", SessionID: tc.saveSessionID, FileName: tc.fileName,
				Part: genai.NewPartFromBytes([]byte("data"), "image/png"), CustomMetadata: metadata,
			}); err != nil {
				t.Fatalf("Save(v1) failed: %v", err)
			}
			metadata["string"] = "changed after save"

			fb.setNow(func() time.Time { return secondCreateTime })
			if _, err := svc.Save(t.Context(), &artifact.SaveRequest{
				AppName: "app", UserID: "user", SessionID: tc.saveSessionID, FileName: tc.fileName,
				Part: genai.NewPartFromText("text"),
			}); err != nil {
				t.Fatalf("Save(v2) failed: %v", err)
			}

			latest, err := svc.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{
				AppName: "app", UserID: "user", SessionID: tc.getSessionID, FileName: tc.fileName,
			})
			if err != nil {
				t.Fatalf("GetArtifactVersion(latest) failed: %v", err)
			}
			wantLatest := &artifact.ArtifactVersion{
				Version:        2,
				CanonicalURI:   tc.uriPrefix + "2",
				CustomMetadata: map[string]any{},
				CreateTime:     secondCreateTime,
				MimeType:       "text/plain",
			}
			if diff := cmp.Diff(wantLatest, latest.ArtifactVersion); diff != "" {
				t.Errorf("GetArtifactVersion(latest) mismatch (-want +got):\n%s", diff)
			}

			first, err := svc.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{
				AppName: "app", UserID: "user", SessionID: tc.getSessionID, FileName: tc.fileName, Version: 1,
			})
			if err != nil {
				t.Fatalf("GetArtifactVersion(v1) failed: %v", err)
			}
			wantFirst := &artifact.ArtifactVersion{
				Version:      1,
				CanonicalURI: tc.uriPrefix + "1",
				CustomMetadata: map[string]any{
					"string":  "value",
					"number":  "42",
					"enabled": "true",
				},
				CreateTime: firstCreateTime,
				MimeType:   "image/png",
			}
			if diff := cmp.Diff(wantFirst, first.ArtifactVersion); diff != "" {
				t.Errorf("GetArtifactVersion(v1) mismatch (-want +got):\n%s", diff)
			}

			first.ArtifactVersion.CustomMetadata["string"] = "changed after read"
			again, err := svc.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{
				AppName: "app", UserID: "user", SessionID: tc.getSessionID, FileName: tc.fileName, Version: 1,
			})
			if err != nil {
				t.Fatalf("second GetArtifactVersion(v1) failed: %v", err)
			}
			if diff := cmp.Diff(wantFirst, again.ArtifactVersion); diff != "" {
				t.Errorf("second GetArtifactVersion(v1) mismatch (-want +got):\n%s", diff)
			}

			if err := svc.Delete(t.Context(), &artifact.DeleteRequest{
				AppName: "app", UserID: "user", SessionID: tc.getSessionID, FileName: tc.fileName,
			}); err != nil {
				t.Fatalf("Delete() failed: %v", err)
			}
			fb.setNow(func() time.Time { return recreateTime })
			if _, err := svc.Save(t.Context(), &artifact.SaveRequest{
				AppName: "app", UserID: "user", SessionID: tc.saveSessionID, FileName: tc.fileName,
				Part: genai.NewPartFromBytes([]byte("new"), "application/octet-stream"),
			}); err != nil {
				t.Fatalf("Save(recreated) failed: %v", err)
			}
			recreated, err := svc.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{
				AppName: "app", UserID: "user", SessionID: tc.getSessionID, FileName: tc.fileName,
			})
			if err != nil {
				t.Fatalf("GetArtifactVersion(recreated) failed: %v", err)
			}
			wantRecreated := &artifact.ArtifactVersion{
				Version:        1,
				CanonicalURI:   tc.uriPrefix + "1",
				CustomMetadata: map[string]any{},
				CreateTime:     recreateTime,
				MimeType:       "application/octet-stream",
			}
			if diff := cmp.Diff(wantRecreated, recreated.ArtifactVersion); diff != "" {
				t.Errorf("GetArtifactVersion(recreated) mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestGCSArtifactServiceConcurrentSave checks that concurrent saves of the same
// artifact get distinct versions and none is overwritten. Pre-fix, colliding
// savers picked the same version and clobbered each other.
func TestGCSArtifactServiceConcurrentSave(t *testing.T) {
	srv, err := newGCSArtifactServiceForTesting("concurrent")
	if err != nil {
		t.Fatalf("failed to set up service: %v", err)
	}

	// Must stay <= maxSaveAttempts: a maximally-unlucky saver can lose one race
	// per other writer, so it may need up to savers attempts to land a version.
	const savers = 10

	var wg sync.WaitGroup
	gotVersions := make([]int64, savers)
	errs := make([]error, savers)
	for i := range savers {
		wg.Go(func() {
			resp, err := srv.Save(t.Context(), saveReq())
			if err != nil {
				errs[i] = err
				return
			}
			gotVersions[i] = resp.Version
		})
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			// Fail fast: a saver error leaves zeroed versions, which would make
			// the version diffs below noisy and misleading.
			t.Fatalf("saver %d: Save() failed: %v", i, err)
		}
	}

	slices.Sort(gotVersions)
	want := make([]int64, savers)
	for i := range want {
		want[i] = int64(i + 1)
	}
	if diff := cmp.Diff(want, gotVersions); diff != "" {
		t.Errorf("returned versions mismatch (-want +got):\n%s", diff)
	}

	// The stored versions must match what Save reported (nothing overwritten).
	resp, err := srv.Versions(t.Context(), &artifact.VersionsRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
	})
	if err != nil {
		t.Fatalf("Versions() failed: %v", err)
	}
	stored := resp.Versions
	slices.Sort(stored)
	if diff := cmp.Diff(want, stored); diff != "" {
		t.Errorf("stored versions mismatch (-want +got):\n%s", diff)
	}
}

// TestSaveReturnsWriteErrorWithoutRetry checks that a non-precondition write
// error is surfaced as-is and not retried.
func TestSaveReturnsWriteErrorWithoutRetry(t *testing.T) {
	svc := newGCSServiceForTesting("errtest")
	fb := svc.bucket.(*fakeBucket)
	wantErr := &googleapi.Error{Code: http.StatusInternalServerError, Message: "boom"}
	fb.closeErr = wantErr

	if _, err := svc.Save(t.Context(), saveReq()); !errors.Is(err, wantErr) {
		t.Fatalf("Save() err = %v, want %v", err, wantErr)
	}
	if got := fb.closeCalls; got != 1 {
		t.Errorf("write attempts = %d, want 1 (no retry on non-precondition error)", got)
	}
}

// TestSaveExhaustsRetriesOnPersistentConflict checks that Save gives up with
// ErrVersionConflict after maxSaveAttempts when every write hits a precondition
// failure.
func TestSaveExhaustsRetriesOnPersistentConflict(t *testing.T) {
	svc := newGCSServiceForTesting("errtest")
	fb := svc.bucket.(*fakeBucket)
	fb.closeErr = &googleapi.Error{Code: http.StatusPreconditionFailed}
	wantMetadata := map[string]string{"key": "value"}
	req := saveReq()
	req.CustomMetadata = map[string]any{"key": "value"}

	if _, err := svc.Save(t.Context(), req); !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("Save() err = %v, want ErrVersionConflict", err)
	}
	if got := fb.closeCalls; got != maxSaveAttempts {
		t.Errorf("write attempts = %d, want %d", got, maxSaveAttempts)
	}
	if got := len(fb.metadataCalls); got != maxSaveAttempts {
		t.Fatalf("SetMetadata calls = %d, want %d", got, maxSaveAttempts)
	}
	for i, got := range fb.metadataCalls {
		if diff := cmp.Diff(wantMetadata, got); diff != "" {
			t.Errorf("SetMetadata call %d mismatch (-want +got):\n%s", i, diff)
		}
	}
}

// TestSaveZeroByteArtifact checks that an empty payload still creates a real,
// listable version. SaveRequest.Validate only requires Part.InlineData, so this
// is reachable through the public API, and treating "no bytes" as "no object"
// would make the second save overwrite the first.
func TestSaveZeroByteArtifact(t *testing.T) {
	svc := newGCSServiceForTesting("zerobyte")
	req := func() *artifact.SaveRequest {
		return &artifact.SaveRequest{
			AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
			Part: genai.NewPartFromBytes([]byte{}, "application/octet-stream"),
		}
	}

	for want := int64(1); want <= 2; want++ {
		resp, err := svc.Save(t.Context(), req())
		if err != nil {
			t.Fatalf("Save() failed: %v", err)
		}
		if resp.Version != want {
			t.Errorf("Save() version = %d, want %d", resp.Version, want)
		}
	}

	versions, err := svc.Versions(t.Context(), &artifact.VersionsRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
	})
	if err != nil {
		t.Fatalf("Versions() failed: %v", err)
	}
	slices.Sort(versions.Versions)
	if diff := cmp.Diff([]int64{1, 2}, versions.Versions); diff != "" {
		t.Errorf("stored versions mismatch (-want +got):\n%s", diff)
	}

	loaded, err := svc.Load(t.Context(), &artifact.LoadRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
	})
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}
	if got := loaded.Part.InlineData.Data; len(got) != 0 {
		t.Errorf("Load() data = %q, want empty", got)
	}
}

// TestNotFoundIsWrappedSentinel checks that Load and GetArtifactVersion
// recognize storage.ErrObjectNotExist even when the storage client wraps it,
// which is how cloud.google.com/go/storage actually returns it in production
// (see storage.formatObjectErr, which always wraps NotFound as
// fmt.Errorf("%w: %w", ErrObjectNotExist, err)). A caller checking
// errors.Is(err, fs.ErrNotExist) must still get a match.
func TestNotFoundIsWrappedSentinel(t *testing.T) {
	wrapped := fmt.Errorf("%w: %w", storage.ErrObjectNotExist, errors.New("googleapi: Error 404: Not Found"))

	for _, tc := range []struct {
		name string
		call func(svc *gcsService) error
	}{
		{
			name: "Load",
			call: func(svc *gcsService) error {
				_, err := svc.Load(t.Context(), &artifact.LoadRequest{
					AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
					Version: 1,
				})
				return err
			},
		},
		{
			name: "GetArtifactVersion",
			call: func(svc *gcsService) error {
				_, err := svc.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{
					AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
					Version: 1,
				})
				return err
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := newGCSServiceForTesting("wrapped-not-exist")
			fb := svc.bucket.(*fakeBucket)
			fb.attrsErr = wrapped

			err := tc.call(svc)
			if !errors.Is(err, fs.ErrNotExist) {
				t.Errorf("err = %v, want errors.Is(err, fs.ErrNotExist) = true", err)
			}
		})
	}
}

// TestGetArtifactVersionCanonicalURI checks the URI handed to a consumer is the
// gs:// form even though the object also has a MediaLink. MediaLink is an
// authenticated JSON API download URL: a model given it as the file_uri of a
// file_data part treats it as a web page and cannot read the object.
func TestGetArtifactVersionCanonicalURI(t *testing.T) {
	for _, tc := range []struct {
		name     string
		fileName string
		wantPath string
	}{
		{name: "session scoped", fileName: "file", wantPath: "app/user/session/file"},
		{name: "user namespaced", fileName: "user:file", wantPath: "app/user/user/user:file"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := newGCSServiceForTesting("demo-bucket")
			saved, err := svc.Save(t.Context(), &artifact.SaveRequest{
				AppName: "app", UserID: "user", SessionID: "session", FileName: tc.fileName,
				Part: genai.NewPartFromBytes([]byte("data"), "text/plain"),
			})
			if err != nil {
				t.Fatalf("Save() failed: %v", err)
			}

			// The assertion below only distinguishes the two forms while the
			// stored object actually has a MediaLink to be preferred over, so
			// pin that rather than leaving it to the fixture.
			blobName := buildBlobName("app", "user", "session", tc.fileName, saved.Version)
			attrs, err := svc.bucket.object(blobName).attrs(t.Context())
			if err != nil {
				t.Fatalf("attrs() failed: %v", err)
			}
			if attrs.MediaLink == "" {
				t.Fatal("stored object has no MediaLink, so this test cannot detect a regression")
			}

			resp, err := svc.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{
				AppName: "app", UserID: "user", SessionID: "session", FileName: tc.fileName,
			})
			if err != nil {
				t.Fatalf("GetArtifactVersion() failed: %v", err)
			}

			want := fmt.Sprintf("gs://demo-bucket/%s/%d", tc.wantPath, saved.Version)
			if got := resp.ArtifactVersion.CanonicalURI; got != want {
				t.Errorf("CanonicalURI = %q, want %q", got, want)
			}
		})
	}
}

func TestGetArtifactVersionCanonicalURIIsStoredObjectName(t *testing.T) {
	for _, tc := range []struct {
		name      string
		appName   string
		userID    string
		sessionID string
		fileName  string
	}{
		{
			name:      "session scoped special characters",
			appName:   "app one",
			userID:    "user/é",
			sessionID: "session%two",
			fileName:  "report 100%.txt",
		},
		{
			name:      "user scoped special characters",
			appName:   "app/one",
			userID:    "user two",
			sessionID: "session three",
			fileName:  "user:résumé %.txt",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := newGCSServiceForTesting("demo-bucket")
			saved, err := svc.Save(t.Context(), &artifact.SaveRequest{
				AppName: tc.appName, UserID: tc.userID, SessionID: tc.sessionID, FileName: tc.fileName,
				Part: genai.NewPartFromText("data"),
			})
			if err != nil {
				t.Fatalf("Save() failed: %v", err)
			}

			resp, err := svc.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{
				AppName: tc.appName, UserID: tc.userID, SessionID: tc.sessionID, FileName: tc.fileName,
			})
			if err != nil {
				t.Fatalf("GetArtifactVersion() failed: %v", err)
			}
			want := fmt.Sprintf("gs://demo-bucket/%s", buildBlobName(tc.appName, tc.userID, tc.sessionID, tc.fileName, saved.Version))
			if got := resp.ArtifactVersion.CanonicalURI; got != want {
				t.Error("CanonicalURI does not name the stored object")
			}
		})
	}
}

// TestBackoffDelayBounds checks the jittered backoff stays within [0, saveRetryMaxDelay].
func TestBackoffDelayBounds(t *testing.T) {
	for attempt := range maxSaveAttempts {
		if d := backoffDelay(attempt); d < 0 || d > saveRetryMaxDelay {
			t.Errorf("backoffDelay(%d) = %v, want within [0, %v]", attempt, d, saveRetryMaxDelay)
		}
	}
}

// TestIsPreconditionFailed covers both transports (HTTP *googleapi.Error and
// gRPC status), raw and wrapped, since Save's retry hinges on this detection.
func TestIsPreconditionFailed(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"generic", errors.New("boom"), false},
		{"http 412", &googleapi.Error{Code: http.StatusPreconditionFailed}, true},
		{"http 412 wrapped", fmt.Errorf("save: %w", &googleapi.Error{Code: http.StatusPreconditionFailed}), true},
		{"http 500", &googleapi.Error{Code: http.StatusInternalServerError}, false},
		{"grpc failed precondition", status.Error(codes.FailedPrecondition, "cond"), true},
		{"grpc failed precondition wrapped", fmt.Errorf("save: %w", status.Error(codes.FailedPrecondition, "cond")), true},
		{"grpc not found", status.Error(codes.NotFound, "nf"), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := isPreconditionFailed(tc.err); got != tc.want {
				t.Errorf("isPreconditionFailed(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

// TestSleepContext checks that sleepContext honors a non-positive duration, an
// already-canceled context, and a normal short wait.
func TestSleepContext(t *testing.T) {
	if err := sleepContext(t.Context(), 0); err != nil {
		t.Errorf("sleepContext(_, 0) = %v, want nil", err)
	}

	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if err := sleepContext(canceled, time.Hour); !errors.Is(err, context.Canceled) {
		t.Errorf("sleepContext(canceled, 1h) = %v, want context.Canceled", err)
	}

	if err := sleepContext(t.Context(), time.Millisecond); err != nil {
		t.Errorf("sleepContext(_, 1ms) = %v, want nil", err)
	}
}

// saveReq returns a minimal valid SaveRequest for the fixed test artifact.
func saveReq() *artifact.SaveRequest {
	return &artifact.SaveRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "file",
		Part: genai.NewPartFromBytes([]byte("data"), "text/plain"),
	}
}

// ---------------------------------- Mock Implementations -----------------------------------
// fakeClient implements the gcsClient interface for testing.
type fakeClient struct {
	inMemoryBucket gcsBucket
}

func newFakeClient() gcsClient {
	return &fakeClient{
		inMemoryBucket: &fakeBucket{
			blobs: make(map[string]*fakeBlob),
			now:   time.Now,
		},
	}
}

// Bucket returns the singleton in-memory bucket.
func (c *fakeClient) bucket(name string) gcsBucket {
	return c.inMemoryBucket
}

// fakeBucket implements the gcsBucket interface for testing.
type fakeBucket struct {
	mu    sync.Mutex
	blobs map[string]*fakeBlob
	now   func() time.Time

	// Test hooks (guarded by mu): closeErr, when set, makes every writer Close
	// return it (a simulated write failure); closeCalls counts Close calls.
	closeErr   error
	closeCalls int
	// metadataCalls records the metadata set on every write attempt.
	metadataCalls []map[string]string

	// attrsErr, when set, is returned by every object's attrs() call in place
	// of the default not-found/found behavior. Used to simulate the GCS
	// client library's wrapped storage.ErrObjectNotExist (see
	// TestNotFoundIsWrappedSentinel).
	attrsErr error

	deleteErr error

	// objectsCalls counts listings, so a test can tell that a request was
	// refused before Save listed the versions.
	objectsCalls int
}

func (f *fakeBucket) setNow(now func() time.Time) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.now = now
}

// object returns a handle to the named blob, creating an empty backing store on
// first use so repeat handles to the same name share state.
func (f *fakeBucket) object(name string) gcsObject {
	f.mu.Lock()
	defer f.mu.Unlock()
	b, ok := f.blobs[name]
	if !ok {
		b = &fakeBlob{name: name}
		f.blobs[name] = b
	}
	return &fakeObject{blob: b, bucket: f}
}

// objects iterates the existing (written, not deleted) blobs matching the
// prefix, snapshotting attributes at call time like a real GCS listing so a
// later delete can't retroactively change what this iterator returns.
func (f *fakeBucket) objects(ctx context.Context, q *storage.Query) gcsObjectIterator {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.objectsCalls++

	var attrs []*storage.ObjectAttrs
	for name, b := range f.blobs {
		if q != nil && q.Prefix != "" && !strings.HasPrefix(name, q.Prefix) {
			continue
		}
		b.mu.Lock()
		if b.exists {
			attrs = append(attrs, &storage.ObjectAttrs{Name: b.name, ContentType: b.contentType})
		}
		b.mu.Unlock()
	}
	return &fakeObjectIterator{attrs: attrs}
}

// signedURL fails: the fake has no credentials, so tests that sign pass a
// URLSigner, and TestDefaultSignerUsesClientCredentials signs with a real
// bucket handle.
func (f *fakeBucket) signedURL(object string, opts *storage.SignedURLOptions) (string, error) {
	return "", errors.New("fake bucket cannot sign URLs")
}

// fakeBlob is the shared backing store for one object name; handles reference it
// by pointer so concurrent writers to the same name exercise the precondition.
type fakeBlob struct {
	mu   sync.Mutex
	name string
	// exists tracks presence separately from data: a handle can be taken
	// without creating an object, and a zero-byte artifact does exist (real GCS
	// creates and lists one).
	exists      bool
	data        []byte
	contentType string
	metadata    map[string]string
	created     time.Time
}

// fakeObject is a handle to a fakeBlob, optionally carrying a does-not-exist
// precondition (mirrors storage.ObjectHandle.If(Conditions{DoesNotExist: true})).
type fakeObject struct {
	blob         *fakeBlob
	bucket       *fakeBucket
	mustNotExist bool
}

// newWriter returns a fake writer that commits data to the blob on Close.
func (o *fakeObject) newWriter(ctx context.Context) gcsWriter {
	return &fakeWriter{obj: o, buffer: &bytes.Buffer{}}
}

func (o *fakeObject) ifNotExist() gcsObject {
	return &fakeObject{blob: o.blob, bucket: o.bucket, mustNotExist: true}
}

// attrs returns fake attributes for the object.
func (o *fakeObject) attrs(ctx context.Context) (*storage.ObjectAttrs, error) {
	if o.bucket != nil {
		o.bucket.mu.Lock()
		forced := o.bucket.attrsErr
		o.bucket.mu.Unlock()
		if forced != nil {
			return nil, forced
		}
	}
	b := o.blob
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.exists {
		return nil, storage.ErrObjectNotExist
	}
	// MediaLink is populated on every object real GCS returns. The fake left it
	// empty, which hid that CanonicalURI preferred it over the gs:// form.
	return &storage.ObjectAttrs{
		Name:        b.name,
		Created:     b.created,
		ContentType: b.contentType,
		Metadata:    maps.Clone(b.metadata),
		MediaLink:   "https://storage.googleapis.com/download/storage/v1/b/bucket/o/" + b.name + "?alt=media",
	}, nil
}

// delete removes the object from the in-memory store.
func (o *fakeObject) delete(ctx context.Context) error {
	if o.bucket != nil {
		o.bucket.mu.Lock()
		forced := o.bucket.deleteErr
		o.bucket.mu.Unlock()
		if forced != nil {
			return forced
		}
	}
	b := o.blob
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.exists {
		return storage.ErrObjectNotExist
	}
	b.exists = false
	b.data = nil
	b.contentType = ""
	b.metadata = nil
	b.created = time.Time{}
	return nil
}

// newReader returns a reader for the in-memory data.
func (o *fakeObject) newReader(ctx context.Context) (io.ReadCloser, error) {
	b := o.blob
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.exists {
		return nil, fs.ErrNotExist
	}
	return io.NopCloser(bytes.NewReader(b.data)), nil
}

// fakeWriter is a helper type to simulate an *storage.Writer.
type fakeWriter struct {
	obj         *fakeObject
	buffer      *bytes.Buffer
	contentType string
	metadata    map[string]string
	started     bool
}

func (w *fakeWriter) Write(p []byte) (n int, err error) {
	w.started = true
	return w.buffer.Write(p)
}

// Close commits the buffered data under the blob lock, so a conditional write
// that loses the race sees the winner's data and returns HTTP 412 like GCS. A
// bucket-level closeErr hook lets tests force a write failure.
func (w *fakeWriter) Close() error {
	now := time.Now
	if bkt := w.obj.bucket; bkt != nil {
		bkt.mu.Lock()
		bkt.closeCalls++
		forced := bkt.closeErr
		now = bkt.now
		bkt.mu.Unlock()
		if forced != nil {
			return forced
		}
	}

	b := w.obj.blob
	b.mu.Lock()
	defer b.mu.Unlock()
	if w.obj.mustNotExist && b.exists {
		return &googleapi.Error{Code: http.StatusPreconditionFailed, Message: "conditionNotMet"}
	}
	b.data = w.buffer.Bytes()
	b.contentType = w.contentType
	b.metadata = maps.Clone(w.metadata)
	b.created = now()
	b.exists = true
	return nil
}

// SetContentType implements the final piece of the interface.
func (w *fakeWriter) SetContentType(cType string) {
	if w.started {
		panic("SetContentType called after Write")
	}
	w.contentType = cType
}

func (w *fakeWriter) SetMetadata(metadata map[string]string) {
	if w.started {
		panic("SetMetadata called after Write")
	}
	w.metadata = maps.Clone(metadata)
	if bkt := w.obj.bucket; bkt != nil {
		bkt.mu.Lock()
		bkt.metadataCalls = append(bkt.metadataCalls, maps.Clone(metadata))
		bkt.mu.Unlock()
	}
}

// fakeObjectIterator returns attribute snapshots taken at list time.
type fakeObjectIterator struct {
	attrs []*storage.ObjectAttrs
	index int
}

// next returns the next object's attributes or an iterator.Done error.
func (i *fakeObjectIterator) next() (*storage.ObjectAttrs, error) {
	if i.index >= len(i.attrs) {
		return nil, iterator.Done
	}
	a := i.attrs[i.index]
	i.index++
	return a, nil
}

var (
	_ gcsClient         = (*fakeClient)(nil)
	_ gcsBucket         = (*fakeBucket)(nil)
	_ gcsObject         = (*fakeObject)(nil)
	_ gcsObjectIterator = (*fakeObjectIterator)(nil)
	_ gcsWriter         = (*fakeWriter)(nil)
)

// TestGCSService_SaveEmptyText checks that GCS stores a part whose every field
// is zero, which SaveRequest.Validate accepts as empty text.
func TestGCSService_SaveEmptyText(t *testing.T) {
	svc := newGCSServiceForTesting("emptytext")
	resp, err := svc.Save(t.Context(), &artifact.SaveRequest{
		AppName: "app", UserID: "user", SessionID: "session", FileName: "empty.txt",
		Part: genai.NewPartFromText(""),
	})
	if err != nil {
		t.Fatalf("Save() failed: %v", err)
	}
	if resp.Version != 1 {
		t.Errorf("Save() version = %d, want 1", resp.Version)
	}
}

// ------------------- Bookkeeping metadata, references and URL lookup -------------------

// putBlob writes an object straight into the fake bucket, as another client
// sharing the bucket (adk-python, or an older adk-go) would.
func putBlob(t *testing.T, svc *gcsService, name string, data []byte, contentType string, metadata map[string]string) {
	t.Helper()
	b := svc.bucket.object(name).(*fakeObject).blob
	b.mu.Lock()
	defer b.mu.Unlock()
	b.exists = true
	b.data = data
	b.contentType = contentType
	b.metadata = maps.Clone(metadata)
	b.created = time.Now()
}

// storedObject is what the fake bucket holds for one object.
type storedObject struct {
	ContentType string
	Metadata    map[string]string
	Data        string
}

func readStored(t *testing.T, svc *gcsService, name string) storedObject {
	t.Helper()
	fb := svc.bucket.(*fakeBucket)
	fb.mu.Lock()
	b, ok := fb.blobs[name]
	fb.mu.Unlock()
	if !ok {
		t.Fatalf("object %q was never written", name)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if !b.exists {
		t.Fatalf("object %q does not exist", name)
	}
	return storedObject{ContentType: b.contentType, Metadata: maps.Clone(b.metadata), Data: string(b.data)}
}

func int64Ptr(v int64) *int64 { return new(v) }

func fileDataPart(uri, mimeType string) *genai.Part {
	return &genai.Part{FileData: &genai.FileData{FileURI: uri, MIMEType: mimeType}}
}

// mustSave saves part as fileName in app/user1/sessionID and returns its version.
func mustSave(t *testing.T, svc *gcsService, sessionID, fileName string, part *genai.Part) int64 {
	t.Helper()
	resp, err := svc.Save(t.Context(), &artifact.SaveRequest{
		AppName: "app", UserID: "user1", SessionID: sessionID, FileName: fileName, Part: part,
	})
	if err != nil {
		t.Fatalf("Save(%q) failed: %v", fileName, err)
	}
	return resp.Version
}

// signerCall is one call a recordingSigner received.
type signerCall struct {
	bucket, object string
	opts           storage.SignedURLOptions
}

// recordingSigner is a URLSigner test double that records its calls and
// returns "signed:<bucket>/<object>", or err when set.
type recordingSigner struct {
	mu    sync.Mutex
	calls []signerCall
	err   error
}

func (r *recordingSigner) sign(bucket, object string, opts *storage.SignedURLOptions) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, signerCall{bucket: bucket, object: object, opts: *opts})
	if r.err != nil {
		return "", r.err
	}
	return "signed:" + bucket + "/" + object, nil
}

func (r *recordingSigner) recorded() []signerCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.calls)
}

// urlLookup runs one of the two URL calls for the artifact named by req.
type urlLookup struct {
	name string
	get  func(ctx context.Context, svc *gcsService, req GetAuthenticatedURLRequest) (string, error)
}

func urlLookups(signer *recordingSigner) []urlLookup {
	return []urlLookup{
		{name: "authenticated", get: func(ctx context.Context, svc *gcsService, req GetAuthenticatedURLRequest) (string, error) {
			return svc.GetAuthenticatedURL(ctx, &req)
		}},
		{name: "signed", get: func(ctx context.Context, svc *gcsService, req GetAuthenticatedURLRequest) (string, error) {
			return svc.GetSignedURL(ctx, &GetSignedURLRequest{
				AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID, FileName: req.FileName,
				Version: req.Version, Signer: signer.sign,
			})
		}},
	}
}

func TestSaveWritesBookkeepingMetadata(t *testing.T) {
	const ref = "artifact://apps/app/users/user1/sessions/sess1/artifacts/t.txt/versions/1"
	for _, tc := range []struct {
		name     string
		part     *genai.Part
		metadata map[string]any
		want     storedObject
	}{
		{
			name: "text",
			part: genai.NewPartFromText("hello"),
			want: storedObject{ContentType: "text/plain", Metadata: map[string]string{"adkIsText": "true"}, Data: "hello"},
		},
		{
			name:     "empty text with caller metadata",
			part:     genai.NewPartFromText(""),
			metadata: map[string]any{"origin": "x"},
			want:     storedObject{ContentType: "text/plain", Metadata: map[string]string{"origin": "x", "adkIsText": "true"}},
		},
		{
			name: "inline data with display name",
			part: &genai.Part{InlineData: &genai.Blob{Data: []byte("abc"), MIMEType: "image/png", DisplayName: "My Report (final).png"}},
			want: storedObject{ContentType: "image/png", Metadata: map[string]string{"adkDisplayName": "My Report (final).png"}, Data: "abc"},
		},
		{
			name: "inline data without display name",
			part: genai.NewPartFromBytes([]byte("abc"), "image/png"),
			want: storedObject{ContentType: "image/png", Data: "abc"},
		},
		{
			name: "inline text/plain data is not flagged as text",
			part: genai.NewPartFromBytes([]byte("x"), "text/plain"),
			want: storedObject{ContentType: "text/plain", Data: "x"},
		},
		{
			name: "external file data",
			part: fileDataPart("gs://my-bucket/report.pdf", "application/pdf"),
			want: storedObject{ContentType: "application/pdf", Metadata: map[string]string{
				"adkFileUri": "gs://my-bucket/report.pdf", "adkFileMimeType": "application/pdf",
			}},
		},
		{
			name: "external file data without MIME type",
			part: fileDataPart("gs://my-bucket/data.bin", ""),
			want: storedObject{Metadata: map[string]string{"adkFileUri": "gs://my-bucket/data.bin"}},
		},
		{
			name: "artifact reference",
			part: fileDataPart(ref, "text/plain"),
			want: storedObject{ContentType: "text/plain", Metadata: map[string]string{
				"adkFileUri": ref, "adkFileMimeType": "text/plain",
			}},
		},
		{
			name: "inline data wins over file data",
			part: &genai.Part{InlineData: &genai.Blob{Data: []byte("abc"), MIMEType: "image/png"}, FileData: &genai.FileData{FileURI: "gs://b/o"}},
			want: storedObject{ContentType: "image/png", Data: "abc"},
		},
		{
			name: "text wins over file data",
			part: &genai.Part{Text: "x", FileData: &genai.FileData{FileURI: "gs://b/o"}},
			want: storedObject{ContentType: "text/plain", Metadata: map[string]string{"adkIsText": "true"}, Data: "x"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := newGCSServiceForTesting("bucket")
			resp, err := svc.Save(t.Context(), &artifact.SaveRequest{
				AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "f.txt",
				Part: tc.part, CustomMetadata: tc.metadata,
			})
			if err != nil {
				t.Fatalf("Save() failed: %v", err)
			}
			if resp.Version != 1 {
				t.Errorf("Save() version = %d, want 1", resp.Version)
			}
			got := readStored(t, svc, "app/user1/sess1/f.txt/1")
			if diff := cmp.Diff(tc.want, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("stored object mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSaveDropsReservedCallerMetadata(t *testing.T) {
	for _, key := range []string{"adkFileUri", "adkFileMimeType", "adkDisplayName", "adkIsText", "file_uri"} {
		t.Run(key, func(t *testing.T) {
			svc := newGCSServiceForTesting("bucket")
			if _, err := svc.Save(t.Context(), &artifact.SaveRequest{
				AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "f.bin",
				Part:           genai.NewPartFromBytes([]byte("REAL"), "application/octet-stream"),
				CustomMetadata: map[string]any{key: "gs://evil/x", "origin": 42},
			}); err != nil {
				t.Fatalf("Save() failed: %v", err)
			}
			got := readStored(t, svc, "app/user1/sess1/f.bin/1").Metadata
			if diff := cmp.Diff(map[string]string{"origin": "42"}, got); diff != "" {
				t.Errorf("stored metadata mismatch (-want +got):\n%s", diff)
			}
			loaded, err := svc.Load(t.Context(), &artifact.LoadRequest{
				AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "f.bin",
			})
			if err != nil {
				t.Fatalf("Load() failed: %v", err)
			}
			want := &genai.Part{InlineData: &genai.Blob{Data: []byte("REAL"), MIMEType: "application/octet-stream"}}
			if diff := cmp.Diff(want, loaded.Part); diff != "" {
				t.Errorf("Load() mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("text flag cannot be overridden", func(t *testing.T) {
		svc := newGCSServiceForTesting("bucket")
		if _, err := svc.Save(t.Context(), &artifact.SaveRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "f.txt",
			Part: genai.NewPartFromText("hi"), CustomMetadata: map[string]any{"adkIsText": "false"},
		}); err != nil {
			t.Fatalf("Save() failed: %v", err)
		}
		got := readStored(t, svc, "app/user1/sess1/f.txt/1").Metadata
		if diff := cmp.Diff(map[string]string{"adkIsText": "true"}, got); diff != "" {
			t.Errorf("stored metadata mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestSaveFileDataValidation(t *testing.T) {
	const prefix = "artifact://apps/app/users/user1/"
	for _, tc := range []struct {
		name       string
		uri        string
		wantPhrase string
	}{
		{name: "empty URI", uri: "", wantPhrase: "file_uri"},
		{name: "malformed reference", uri: "artifact://apps/app/invalid", wantPhrase: "Invalid artifact reference URI"},
		{name: "other app", uri: "artifact://apps/other/users/user1/sessions/sess1/artifacts/t.txt/versions/1", wantPhrase: "same app and user scope"},
		{name: "other user", uri: "artifact://apps/app/users/other/sessions/sess1/artifacts/t.txt/versions/1", wantPhrase: "same app and user scope"},
		{name: "other session", uri: prefix + "sessions/other/artifacts/t.txt/versions/1", wantPhrase: "same session scope"},
		{name: "version zero", uri: prefix + "sessions/sess1/artifacts/t.txt/versions/0", wantPhrase: "Invalid artifact reference URI"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := newGCSServiceForTesting("bucket")
			fb := svc.bucket.(*fakeBucket)
			_, err := svc.Save(t.Context(), &artifact.SaveRequest{
				AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "f.txt",
				Part: fileDataPart(tc.uri, "text/plain"),
			})
			if !errors.Is(err, artifact.ErrInputValidation) {
				t.Fatalf("Save() error = %v, want ErrInputValidation", err)
			}
			if !strings.Contains(err.Error(), tc.wantPhrase) {
				t.Errorf("Save() error = %q, want it to contain %q", err, tc.wantPhrase)
			}
			fb.mu.Lock()
			listed, written := fb.objectsCalls, fb.closeCalls
			fb.mu.Unlock()
			if listed != 0 || written != 0 {
				t.Errorf("Save() listed %d times and wrote %d times before refusing, want 0 and 0", listed, written)
			}
		})
	}

	t.Run("message leaves the URI out", func(t *testing.T) {
		svc := newGCSServiceForTesting("bucket")
		_, err := svc.Save(t.Context(), &artifact.SaveRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "f.txt",
			Part: fileDataPart("artifact://apps/zz-secret-zz/users/user1/sessions/sess1/artifacts/t.txt/versions/1", ""),
		})
		if err == nil || strings.Contains(err.Error(), "zz-secret-zz") {
			t.Errorf("Save() error = %v, want an error without the URI", err)
		}
	})

	for _, tc := range []struct {
		name      string
		sessionID string
		uri       string
	}{
		{name: "same-session reference", sessionID: "sess1", uri: prefix + "sessions/sess1/artifacts/t.txt/versions/1"},
		{name: "user-scoped reference from another session", sessionID: "sess2", uri: prefix + "artifacts/user:n.txt/versions/1"},
		{name: "external file data", sessionID: "sess1", uri: "gs://b/o"},
		{name: "artifact scheme without slashes is external file data", sessionID: "sess1", uri: "artifact:notes"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := newGCSServiceForTesting("bucket")
			if got := mustSave(t, svc, tc.sessionID, "f.txt", fileDataPart(tc.uri, "")); got != 1 {
				t.Errorf("Save() version = %d, want 1", got)
			}
		})
	}
}

func TestLoadRestoresPartFromMetadata(t *testing.T) {
	for _, tc := range []struct {
		name        string
		data        string
		contentType string
		metadata    map[string]string
		want        *genai.Part
	}{
		{
			name: "text flag compares exactly", data: "hi", contentType: "text/plain",
			metadata: map[string]string{"adkIsText": "True"},
			want:     &genai.Part{InlineData: &genai.Blob{Data: []byte("hi"), MIMEType: "text/plain"}},
		},
		{
			name: "display name", data: "\x01\x02\x03", contentType: "image/png",
			metadata: map[string]string{"adkDisplayName": "My Report (final).png"},
			want:     &genai.Part{InlineData: &genai.Blob{Data: []byte{1, 2, 3}, MIMEType: "image/png", DisplayName: "My Report (final).png"}},
		},
		{
			name:     "legacy file_uri",
			metadata: map[string]string{"file_uri": "gs://old/obj"},
			want:     fileDataPart("gs://old/obj", ""),
		},
		{
			name:     "empty adkFileUri falls through to file_uri",
			metadata: map[string]string{"adkFileUri": "", "file_uri": "gs://old/obj"},
			want:     fileDataPart("gs://old/obj", ""),
		},
		{
			name:     "adkFileUri wins over file_uri",
			metadata: map[string]string{"adkFileUri": "gs://new/o", "file_uri": "gs://old/obj"},
			want:     fileDataPart("gs://new/o", ""),
		},
		{
			name: "file URI wins over the text flag", data: "x",
			metadata: map[string]string{"adkFileUri": "gs://b/o", "adkIsText": "true"},
			want:     fileDataPart("gs://b/o", ""),
		},
		{
			name: "text flag wins over the display name", data: "x", contentType: "text/plain",
			metadata: map[string]string{"adkIsText": "true", "adkDisplayName": "n"},
			want:     &genai.Part{Text: "x"},
		},
		{
			name: "MIME type key wins over the content type", contentType: "text/plain; charset=utf-8",
			metadata: map[string]string{"adkFileUri": "gs://b/o", "adkFileMimeType": "application/pdf"},
			want:     fileDataPart("gs://b/o", "application/pdf"),
		},
		{
			name: "empty MIME type key wins over the content type", contentType: "application/pdf",
			metadata: map[string]string{"adkFileUri": "gs://b/o", "adkFileMimeType": ""},
			want:     fileDataPart("gs://b/o", ""),
		},
		{
			name: "content type when no MIME type key", contentType: "application/pdf",
			metadata: map[string]string{"file_uri": "gs://old/obj"},
			want:     fileDataPart("gs://old/obj", "application/pdf"),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := newGCSServiceForTesting("bucket")
			putBlob(t, svc, "app/user1/sess1/f.txt/1", []byte(tc.data), tc.contentType, tc.metadata)
			got, err := svc.Load(t.Context(), &artifact.LoadRequest{
				AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "f.txt", Version: 1,
			})
			if err != nil {
				t.Fatalf("Load() failed: %v", err)
			}
			if diff := cmp.Diff(tc.want, got.Part); diff != "" {
				t.Errorf("Load() mismatch (-want +got):\n%s", diff)
			}
		})
	}

	for _, tc := range []struct {
		name string
		part *genai.Part
	}{
		{name: "multibyte text", part: genai.NewPartFromText("héllo wörld")},
		{name: "inline data with display name", part: &genai.Part{InlineData: &genai.Blob{
			Data: []byte("abc"), MIMEType: "image/png", DisplayName: "My Report (final).png",
		}}},
		{name: "inline data without display name", part: genai.NewPartFromBytes([]byte("abc"), "image/png")},
	} {
		t.Run("round trip/"+tc.name, func(t *testing.T) {
			svc := newGCSServiceForTesting("bucket")
			mustSave(t, svc, "sess1", "f.txt", tc.part)
			got, err := svc.Load(t.Context(), &artifact.LoadRequest{
				AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "f.txt",
			})
			if err != nil {
				t.Fatalf("Load() failed: %v", err)
			}
			if diff := cmp.Diff(tc.part, got.Part); diff != "" {
				t.Errorf("Load() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGetArtifactVersionHidesReservedMetadata(t *testing.T) {
	reserved := map[string]string{
		"adkIsText": "true", "adkDisplayName": "n", "adkFileUri": "gs://b/o",
		"adkFileMimeType": "application/pdf", "file_uri": "gs://old/obj",
	}
	withOrigin := maps.Clone(reserved)
	withOrigin["origin"] = "unit-test"
	getVersion := func(t *testing.T, svc *gcsService, fileName string) *artifact.ArtifactVersion {
		t.Helper()
		resp, err := svc.GetArtifactVersion(t.Context(), &artifact.GetArtifactVersionRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: fileName,
		})
		if err != nil {
			t.Fatalf("GetArtifactVersion(%q) failed: %v", fileName, err)
		}
		return resp.ArtifactVersion
	}

	svc := newGCSServiceForTesting("bucket")
	putBlob(t, svc, "app/user1/sess1/mixed.txt/1", nil, "", withOrigin)
	putBlob(t, svc, "app/user1/sess1/reserved.txt/1", nil, "", reserved)
	if _, err := svc.Save(t.Context(), &artifact.SaveRequest{
		AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "ref.pdf",
		Part: fileDataPart("gs://b/report.pdf", "application/pdf"), CustomMetadata: map[string]any{"team": "a"},
	}); err != nil {
		t.Fatalf("Save() failed: %v", err)
	}

	if diff := cmp.Diff(map[string]any{"origin": "unit-test"}, getVersion(t, svc, "mixed.txt").CustomMetadata); diff != "" {
		t.Errorf("CustomMetadata of a mixed object mismatch (-want +got):\n%s", diff)
	}
	if got := getVersion(t, svc, "reserved.txt").CustomMetadata; got == nil || len(got) != 0 {
		t.Errorf("CustomMetadata of a bookkeeping-only object = %#v, want an empty non-nil map", got)
	}
	got := getVersion(t, svc, "ref.pdf")
	if diff := cmp.Diff(map[string]any{"team": "a"}, got.CustomMetadata); diff != "" {
		t.Errorf("CustomMetadata of file data mismatch (-want +got):\n%s", diff)
	}
	if got.MimeType != "application/pdf" {
		t.Errorf("MimeType of file data = %q, want %q", got.MimeType, "application/pdf")
	}
}

func TestVersionsSortedAndIgnoresStrayObjects(t *testing.T) {
	svc := newGCSServiceForTesting("bucket")
	for i := 1; i <= 12; i++ {
		mustSave(t, svc, "sess1", "notes.txt", genai.NewPartFromText(fmt.Sprintf("v%d", i)))
	}
	strays := []string{"checkpoint", "sub/3", "+13", "-1", "٣", "", "99999999999999999999"}
	for _, stray := range strays {
		putBlob(t, svc, "app/user1/sess1/notes.txt/"+stray, []byte("stray"), "text/plain", nil)
		putBlob(t, svc, "app/user1/sess1/strays.txt/"+stray, []byte("stray"), "text/plain", nil)
	}

	got, err := svc.Versions(t.Context(), &artifact.VersionsRequest{
		AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "notes.txt",
	})
	if err != nil {
		t.Fatalf("Versions() failed: %v", err)
	}
	if diff := cmp.Diff([]int64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, got.Versions); diff != "" {
		t.Errorf("Versions() mismatch (-want +got):\n%s", diff)
	}
	latest, err := svc.Load(t.Context(), &artifact.LoadRequest{
		AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "notes.txt",
	})
	if err != nil {
		t.Fatalf("Load(latest) failed: %v", err)
	}
	if latest.Part.Text != "v12" {
		t.Errorf("Load(latest) text = %q, want %q", latest.Part.Text, "v12")
	}
	if got := mustSave(t, svc, "sess1", "notes.txt", genai.NewPartFromText("v13")); got != 13 {
		t.Errorf("Save() after 12 versions = %d, want 13", got)
	}
	if _, err := svc.Versions(t.Context(), &artifact.VersionsRequest{
		AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "strays.txt",
	}); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("Versions(only strays) error = %v, want fs.ErrNotExist", err)
	}
}

// refTo returns a reference part to version 1 of fileName in app/user1/sess1.
func refTo(fileName string) *genai.Part {
	return fileDataPart("artifact://apps/app/users/user1/sessions/sess1/artifacts/"+fileName+"/versions/1", "")
}

func TestLoadReferenceDepth(t *testing.T) {
	svc := newGCSServiceForTesting("bucket")
	mustSave(t, svc, "sess1", "r0.txt", genai.NewPartFromText("end"))
	for i := 1; i <= 6; i++ {
		mustSave(t, svc, "sess1", fmt.Sprintf("r%d.txt", i), refTo(fmt.Sprintf("r%d.txt", i-1)))
	}
	mustSave(t, svc, "sess1", "loop.txt", refTo("loop.txt"))
	load := func(fileName string) (*artifact.LoadResponse, error) {
		return svc.Load(t.Context(), &artifact.LoadRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: fileName,
		})
	}

	got, err := load("r5.txt")
	if err != nil {
		t.Fatalf("Load(5 references) failed: %v", err)
	}
	if diff := cmp.Diff(&genai.Part{Text: "end"}, got.Part); diff != "" {
		t.Errorf("Load(5 references) mismatch (-want +got):\n%s", diff)
	}
	for _, fileName := range []string{"r6.txt", "loop.txt"} {
		_, err := load(fileName)
		if !errors.Is(err, artifact.ErrInputValidation) || !strings.Contains(err.Error(), "Exceeded maximum recursion depth") {
			t.Errorf("Load(%q) error = %v, want ErrInputValidation with %q", fileName, err, "Exceeded maximum recursion depth")
		}
		if err != nil && strings.Contains(err.Error(), "artifact://") {
			t.Errorf("Load(%q) error = %q, want no URI in it", fileName, err)
		}
	}
}

func TestLoadReferenceRevalidatesScopeOnEveryHop(t *testing.T) {
	for _, tc := range []struct {
		name       string
		uri        string
		wantPhrase string
	}{
		{name: "other session", uri: "artifact://apps/app/users/user1/sessions/other/artifacts/t.txt/versions/1", wantPhrase: "same session scope"},
		{name: "other app", uri: "artifact://apps/other/users/user1/sessions/sess1/artifacts/t.txt/versions/1", wantPhrase: "same app and user scope"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			svc := newGCSServiceForTesting("bucket")
			putBlob(t, svc, "app/user1/sess1/ref.txt/1", nil, "", map[string]string{"adkFileUri": tc.uri})
			_, err := svc.Load(t.Context(), &artifact.LoadRequest{
				AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "ref.txt",
			})
			if !errors.Is(err, artifact.ErrInputValidation) || !strings.Contains(err.Error(), tc.wantPhrase) {
				t.Errorf("Load() error = %v, want ErrInputValidation with %q", err, tc.wantPhrase)
			}
		})
	}

	// A hop through a user-scoped URI carries no session, so the session
	// reference it holds fails the scope check even though it names the
	// session of the first caller.
	t.Run("session hop after a user-scoped hop", func(t *testing.T) {
		svc := newGCSServiceForTesting("bucket")
		mustSave(t, svc, "sess1", "t.txt", genai.NewPartFromText("t"))
		mustSave(t, svc, "sess1", "user:bridge.txt", refTo("t.txt"))
		mustSave(t, svc, "sess1", "ref.txt", fileDataPart("artifact://apps/app/users/user1/artifacts/user:bridge.txt/versions/1", ""))
		_, err := svc.Load(t.Context(), &artifact.LoadRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "ref.txt",
		})
		if !errors.Is(err, artifact.ErrInputValidation) || !strings.Contains(err.Error(), "same session scope") {
			t.Errorf("Load() error = %v, want ErrInputValidation with %q", err, "same session scope")
		}
	})

	t.Run("second hop", func(t *testing.T) {
		svc := newGCSServiceForTesting("bucket")
		mustSave(t, svc, "sess1", "t.txt", genai.NewPartFromText("t"))
		mustSave(t, svc, "sess1", "ref2.txt", refTo("t.txt"))
		mustSave(t, svc, "sess1", "ref1.txt", refTo("ref2.txt"))
		putBlob(t, svc, "app/user1/sess1/ref2.txt/1", nil, "", map[string]string{
			"adkFileUri": "artifact://apps/app/users/other/sessions/sess1/artifacts/t.txt/versions/1",
		})
		putBlob(t, svc, "app/other/sess1/t.txt/1", []byte("leaked"), "text/plain", map[string]string{"adkIsText": "true"})
		_, err := svc.Load(t.Context(), &artifact.LoadRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "ref1.txt",
		})
		if !errors.Is(err, artifact.ErrInputValidation) || !strings.Contains(err.Error(), "same app and user scope") {
			t.Errorf("Load() error = %v, want ErrInputValidation with %q", err, "same app and user scope")
		}
	})
}

func TestLoadReferenceTargets(t *testing.T) {
	svc := newGCSServiceForTesting("bucket")
	mustSave(t, svc, "sess1", "missing.txt", fileDataPart("artifact://apps/app/users/user1/sessions/sess1/artifacts/t.txt/versions/9", ""))
	mustSave(t, svc, "sess1", "ext.pdf", fileDataPart("gs://b/o", "application/pdf"))
	mustSave(t, svc, "sess1", "extref.txt", refTo("ext.pdf"))
	mustSave(t, svc, "sess1", "colon.txt", fileDataPart("artifact:notes", ""))
	mustSave(t, svc, "sess2", "user:u.txt", genai.NewPartFromText("mine"))
	mustSave(t, svc, "sess1", "userref.txt", fileDataPart("artifact://apps/app/users/user1/artifacts/user:u.txt/versions/1", ""))
	putBlob(t, svc, "app/user1/sess1/noscope.txt/1", nil, "", map[string]string{
		"adkFileUri": "artifact://apps/app/users/user1/artifacts/plain.txt/versions/1",
	})
	// What a hop without a session would read if it did not validate the target.
	putBlob(t, svc, "app/user1//plain.txt/1", []byte("leaked"), "text/plain", map[string]string{"adkIsText": "true"})
	load := func(fileName string) (*artifact.LoadResponse, error) {
		return svc.Load(t.Context(), &artifact.LoadRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: fileName,
		})
	}

	if _, err := load("missing.txt"); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("Load(reference to a missing version) error = %v, want fs.ErrNotExist", err)
	}
	for _, tc := range []struct {
		fileName string
		want     *genai.Part
	}{
		{fileName: "extref.txt", want: fileDataPart("gs://b/o", "application/pdf")},
		{fileName: "colon.txt", want: fileDataPart("artifact:notes", "")},
		{fileName: "userref.txt", want: &genai.Part{Text: "mine"}},
	} {
		got, err := load(tc.fileName)
		if err != nil {
			t.Errorf("Load(%q) failed: %v", tc.fileName, err)
			continue
		}
		if diff := cmp.Diff(tc.want, got.Part); diff != "" {
			t.Errorf("Load(%q) mismatch (-want +got):\n%s", tc.fileName, diff)
		}
	}
	if _, err := load("noscope.txt"); !errors.Is(err, artifact.ErrInputValidation) {
		t.Errorf("Load(user-scoped URI naming a session file) error = %v, want ErrInputValidation", err)
	}
}

// A reference to version 1 of a target that also has a version 2 reads
// version 1, in Load and in both URL calls.
func TestReferenceReadsTheNamedVersion(t *testing.T) {
	svc := newGCSServiceForTesting("bucket")
	mustSave(t, svc, "sess1", "src.txt", genai.NewPartFromText("first"))
	mustSave(t, svc, "sess1", "src.txt", genai.NewPartFromText("second"))
	mustSave(t, svc, "sess1", "ref.txt", refTo("src.txt"))

	t.Run("load", func(t *testing.T) {
		got, err := svc.Load(t.Context(), &artifact.LoadRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "ref.txt",
		})
		if err != nil {
			t.Fatalf("Load() failed: %v", err)
		}
		if diff := cmp.Diff(&genai.Part{Text: "first"}, got.Part); diff != "" {
			t.Errorf("Load() mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("authenticated", func(t *testing.T) {
		got, err := svc.GetAuthenticatedURL(t.Context(), &GetAuthenticatedURLRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "ref.txt",
		})
		if want := "https://storage.cloud.google.com/bucket/app/user1/sess1/src.txt/1"; got != want || err != nil {
			t.Errorf("GetAuthenticatedURL() = (%q, %v), want (%q, nil)", got, err, want)
		}
	})

	t.Run("signed", func(t *testing.T) {
		signer := &recordingSigner{}
		got, err := svc.GetSignedURL(t.Context(), &GetSignedURLRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "ref.txt", Signer: signer.sign,
		})
		if want := "signed:bucket/app/user1/sess1/src.txt/1"; got != want || err != nil {
			t.Errorf("GetSignedURL() = (%q, %v), want (%q, nil)", got, err, want)
		}
	})
}

func TestGetAuthenticatedURLEncoding(t *testing.T) {
	for _, tc := range []struct {
		sessionID, fileName, want string
	}{
		{"sess1", "é 日.txt", "https://storage.cloud.google.com/bucket/app/user1/sess1/%C3%A9%20%E6%97%A5.txt/1"},
		{"sess1", "a+b'c*d!(e)[f]{g}@h&i=j,k;l$m.txt", "https://storage.cloud.google.com/bucket/app/user1/sess1/a%2Bb%27c%2Ad%21%28e%29%5Bf%5D%7Bg%7D%40h%26i%3Dj%2Ck%3Bl%24m.txt/1"},
		{"sess1", "~-_.AZaz09", "https://storage.cloud.google.com/bucket/app/user1/sess1/~-_.AZaz09/1"},
		{"sess1", "tab\there\x7f.txt", "https://storage.cloud.google.com/bucket/app/user1/sess1/tab%09here%7F.txt/1"},
		{"sess1", "100%.txt", "https://storage.cloud.google.com/bucket/app/user1/sess1/100%25.txt/1"},
		{"sess1", "user:profile.png", "https://storage.cloud.google.com/bucket/app/user1/user/user%3Aprofile.png/1"},
		{"sess1", "notes#1?v=100%real name.txt", "https://storage.cloud.google.com/bucket/app/user1/sess1/notes%231%3Fv%3D100%25real%20name.txt/1"},
		{"sess/1", "f.txt", "https://storage.cloud.google.com/bucket/app/user1/sess/1/f.txt/1"},
	} {
		t.Run(tc.fileName, func(t *testing.T) {
			svc := newGCSServiceForTesting("bucket")
			mustSave(t, svc, tc.sessionID, tc.fileName, genai.NewPartFromText("x"))
			got, err := svc.GetAuthenticatedURL(t.Context(), &GetAuthenticatedURLRequest{
				AppName: "app", UserID: "user1", SessionID: tc.sessionID, FileName: tc.fileName,
			})
			if err != nil {
				t.Fatalf("GetAuthenticatedURL() failed: %v", err)
			}
			if got != tc.want {
				t.Errorf("GetAuthenticatedURL() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestURLLookupVersionSelection(t *testing.T) {
	svc := newGCSServiceForTesting("bucket")
	for i := 1; i <= 12; i++ {
		mustSave(t, svc, "sess1", "n.txt", genai.NewPartFromText(fmt.Sprintf("v%d", i)))
	}
	mustSave(t, svc, "sess1", "user:f.txt", genai.NewPartFromText("u"))

	for _, tc := range []struct {
		name       string
		req        GetAuthenticatedURLRequest
		wantAuth   string
		wantSigned string
	}{
		{
			name:       "latest is the highest version",
			req:        GetAuthenticatedURLRequest{SessionID: "sess1", FileName: "n.txt"},
			wantAuth:   "https://storage.cloud.google.com/bucket/app/user1/sess1/n.txt/12",
			wantSigned: "signed:bucket/app/user1/sess1/n.txt/12",
		},
		{
			name:       "explicit version",
			req:        GetAuthenticatedURLRequest{SessionID: "sess1", FileName: "n.txt", Version: int64Ptr(3)},
			wantAuth:   "https://storage.cloud.google.com/bucket/app/user1/sess1/n.txt/3",
			wantSigned: "signed:bucket/app/user1/sess1/n.txt/3",
		},
		{
			name: "missing explicit version",
			req:  GetAuthenticatedURLRequest{SessionID: "sess1", FileName: "n.txt", Version: int64Ptr(13)},
		},
		{
			name: "version zero is a version, not the latest",
			req:  GetAuthenticatedURLRequest{SessionID: "sess1", FileName: "n.txt", Version: int64Ptr(0)},
		},
		{
			name:       "user-scoped name ignores the session",
			req:        GetAuthenticatedURLRequest{SessionID: "other-session", FileName: "user:f.txt"},
			wantAuth:   "https://storage.cloud.google.com/bucket/app/user1/user/user%3Af.txt/1",
			wantSigned: "signed:bucket/app/user1/user/user:f.txt/1",
		},
	} {
		tc.req.AppName, tc.req.UserID = "app", "user1"
		for _, lookup := range urlLookups(&recordingSigner{}) {
			t.Run(lookup.name+"/"+tc.name, func(t *testing.T) {
				want := tc.wantAuth
				if lookup.name == "signed" {
					want = tc.wantSigned
				}
				got, err := lookup.get(t.Context(), svc, tc.req)
				if err != nil {
					t.Fatalf("URL lookup failed: %v", err)
				}
				if got != want {
					t.Errorf("URL lookup = %q, want %q", got, want)
				}
			})
		}
	}
}

func TestURLLookupStorageErrors(t *testing.T) {
	for _, tc := range []struct {
		name     string
		attrsErr error
		wantErr  error
	}{
		{name: "permission denied", attrsErr: fs.ErrPermission, wantErr: fs.ErrPermission},
		{name: "canceled", attrsErr: context.Canceled, wantErr: context.Canceled},
		{name: "wrapped not found", attrsErr: fmt.Errorf("x: %w", storage.ErrObjectNotExist)},
	} {
		signer := &recordingSigner{}
		for _, lookup := range urlLookups(signer) {
			t.Run(lookup.name+"/"+tc.name, func(t *testing.T) {
				svc := newGCSServiceForTesting("bucket")
				mustSave(t, svc, "sess1", "f.txt", genai.NewPartFromText("x"))
				fb := svc.bucket.(*fakeBucket)
				fb.mu.Lock()
				fb.attrsErr = tc.attrsErr
				fb.mu.Unlock()

				got, err := lookup.get(t.Context(), svc, GetAuthenticatedURLRequest{
					AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "f.txt",
				})
				if got != "" {
					t.Errorf("URL lookup = %q, want empty", got)
				}
				if tc.wantErr == nil && err != nil {
					t.Errorf("URL lookup error = %v, want nil", err)
				}
				if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
					t.Errorf("URL lookup error = %v, want %v", err, tc.wantErr)
				}
				if calls := signer.recorded(); len(calls) != 0 {
					t.Errorf("signer called %d times, want 0", len(calls))
				}
			})
		}
	}
}

func TestURLLookupRequestValidation(t *testing.T) {
	signer := &recordingSigner{}
	for _, lookup := range urlLookups(signer) {
		for _, tc := range []struct {
			name string
			req  GetAuthenticatedURLRequest
		}{
			{name: "empty app", req: GetAuthenticatedURLRequest{UserID: "user1", SessionID: "sess1", FileName: "f.txt"}},
			{name: "empty user", req: GetAuthenticatedURLRequest{AppName: "app", SessionID: "sess1", FileName: "f.txt"}},
			{name: "empty file name", req: GetAuthenticatedURLRequest{AppName: "app", UserID: "user1", SessionID: "sess1"}},
			{name: "session-scoped name without session", req: GetAuthenticatedURLRequest{AppName: "app", UserID: "user1", FileName: "f.txt"}},
			{name: "traversal app", req: GetAuthenticatedURLRequest{AppName: "..", UserID: "user1", SessionID: "sess1", FileName: "f.txt"}},
			{name: "file name with separator", req: GetAuthenticatedURLRequest{AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "a/b"}},
		} {
			// An explicit version skips the version listing, so it shows that
			// the request itself is validated.
			for _, version := range []*int64{nil, int64Ptr(1)} {
				req := tc.req
				req.Version = version
				t.Run(fmt.Sprintf("%s/%s/explicit version %t", lookup.name, tc.name, version != nil), func(t *testing.T) {
					svc := newGCSServiceForTesting("bucket")
					got, err := lookup.get(t.Context(), svc, req)
					if got != "" || !errors.Is(err, artifact.ErrInputValidation) {
						t.Errorf("URL lookup = (%q, %v), want (\"\", ErrInputValidation)", got, err)
					}
				})
			}
		}
		for _, sessionID := range []string{"", "../x"} {
			t.Run(lookup.name+"/user-scoped name with session "+sessionID, func(t *testing.T) {
				svc := newGCSServiceForTesting("bucket")
				got, err := lookup.get(t.Context(), svc, GetAuthenticatedURLRequest{
					AppName: "app", UserID: "user1", SessionID: sessionID, FileName: "user:f.txt",
				})
				if got != "" || err != nil {
					t.Errorf("URL lookup = (%q, %v), want (\"\", nil)", got, err)
				}
			})
		}
	}
	if calls := signer.recorded(); len(calls) != 0 {
		t.Errorf("signer called %d times, want 0", len(calls))
	}
}

func TestURLLookupFollowsReferences(t *testing.T) {
	svc := newGCSServiceForTesting("bucket")
	mustSave(t, svc, "sess1", "src.txt", genai.NewPartFromText("s"))
	mustSave(t, svc, "sess1", "ref1.txt", refTo("src.txt"))
	mustSave(t, svc, "sess1", "ref2.txt", refTo("ref1.txt"))
	mustSave(t, svc, "sess1", "missing.txt", fileDataPart("artifact://apps/app/users/user1/sessions/sess1/artifacts/src.txt/versions/9", ""))
	mustSave(t, svc, "sess1", "ext.pdf", fileDataPart("gs://b/o", "application/pdf"))
	mustSave(t, svc, "sess1", "extref.txt", refTo("ext.pdf"))
	mustSave(t, svc, "sess1", "web.pdf", fileDataPart("https://example.com/a.pdf", "application/pdf"))
	mustSave(t, svc, "sess1", "c1.txt", refTo("src.txt"))
	for i := 2; i <= 6; i++ {
		mustSave(t, svc, "sess1", fmt.Sprintf("c%d.txt", i), refTo(fmt.Sprintf("c%d.txt", i-1)))
	}
	putBlob(t, svc, "app/user1/sess1/cross.txt/1", nil, "", map[string]string{
		"adkFileUri": "artifact://apps/app/users/user1/sessions/other/artifacts/src.txt/versions/1",
	})
	putBlob(t, svc, "app/user1/sess1/bad.txt/1", nil, "", map[string]string{"adkFileUri": "artifact://nonsense"})

	const srcBlob = "app/user1/sess1/src.txt/1"
	for _, tc := range []struct {
		fileName   string
		wantTarget bool
		wantPhrase string
	}{
		{fileName: "ref1.txt", wantTarget: true},
		{fileName: "ref2.txt", wantTarget: true},
		{fileName: "missing.txt"},
		{fileName: "extref.txt"},
		{fileName: "web.pdf"},
		{fileName: "c5.txt", wantTarget: true},
		{fileName: "c6.txt", wantPhrase: "Exceeded maximum recursion depth"},
		{fileName: "cross.txt", wantPhrase: "same session scope"},
		{fileName: "bad.txt", wantPhrase: "Invalid artifact reference URI"},
	} {
		check := func(t *testing.T, got string, err error, wantURL string) {
			t.Helper()
			if tc.wantPhrase != "" {
				if got != "" || !errors.Is(err, artifact.ErrInputValidation) || !strings.Contains(err.Error(), tc.wantPhrase) {
					t.Errorf("URL lookup = (%q, %v), want ErrInputValidation with %q", got, err, tc.wantPhrase)
				}
				return
			}
			if err != nil {
				t.Fatalf("URL lookup failed: %v", err)
			}
			if !tc.wantTarget {
				wantURL = ""
			}
			if got != wantURL {
				t.Errorf("URL lookup = %q, want %q", got, wantURL)
			}
		}

		t.Run("authenticated/"+tc.fileName, func(t *testing.T) {
			got, err := svc.GetAuthenticatedURL(t.Context(), &GetAuthenticatedURLRequest{
				AppName: "app", UserID: "user1", SessionID: "sess1", FileName: tc.fileName,
			})
			check(t, got, err, "https://storage.cloud.google.com/bucket/"+srcBlob)
		})

		t.Run("signed/"+tc.fileName, func(t *testing.T) {
			signer := &recordingSigner{}
			before := time.Now()
			got, err := svc.GetSignedURL(t.Context(), &GetSignedURLRequest{
				AppName: "app", UserID: "user1", SessionID: "sess1", FileName: tc.fileName,
				Expiration: 3 * time.Hour, Method: "PUT", Scheme: storage.SigningSchemeV2, Signer: signer.sign,
			})
			after := time.Now()
			check(t, got, err, "signed:bucket/"+srcBlob)
			calls := signer.recorded()
			if !tc.wantTarget {
				if len(calls) != 0 {
					t.Errorf("signer called %d times, want 0", len(calls))
				}
				return
			}
			if len(calls) != 1 {
				t.Fatalf("signer called %d times, want 1", len(calls))
			}
			c := calls[0]
			if c.object != srcBlob || c.opts.Method != "PUT" || c.opts.Scheme != storage.SigningSchemeV2 {
				t.Errorf("signer got object %q, method %q, scheme %v; want %q, PUT, V2", c.object, c.opts.Method, c.opts.Scheme, srcBlob)
			}
			if c.opts.Expires.Before(before.Add(3*time.Hour)) || c.opts.Expires.After(after.Add(3*time.Hour)) {
				t.Errorf("signer got Expires %v, want 3h after the call", c.opts.Expires)
			}
		})
	}
}

func TestGetSignedURLOptions(t *testing.T) {
	const object = "app/user1/user/user:profile.png/1"
	svc := newGCSServiceForTesting("bucket")
	mustSave(t, svc, "sess1", "user:profile.png", genai.NewPartFromText("p"))
	mustSave(t, svc, "sess1", "ext.pdf", fileDataPart("gs://b/o", "application/pdf"))

	for _, tc := range []struct {
		name       string
		req        GetSignedURLRequest
		wantMethod string
		wantScheme storage.SigningScheme
		wantExpiry time.Duration
	}{
		{name: "defaults", wantMethod: "GET", wantScheme: storage.SigningSchemeDefault, wantExpiry: time.Hour},
		{name: "expiration", req: GetSignedURLRequest{Expiration: 2 * time.Hour}, wantMethod: "GET", wantScheme: storage.SigningSchemeDefault, wantExpiry: 2 * time.Hour},
		{name: "method and scheme", req: GetSignedURLRequest{Method: "PUT", Scheme: storage.SigningSchemeV4}, wantMethod: "PUT", wantScheme: storage.SigningSchemeV4, wantExpiry: time.Hour},
	} {
		t.Run(tc.name, func(t *testing.T) {
			signer := &recordingSigner{}
			req := tc.req
			req.AppName, req.UserID, req.SessionID, req.FileName, req.Signer = "app", "user1", "sess1", "user:profile.png", signer.sign
			before := time.Now()
			got, err := svc.GetSignedURL(t.Context(), &req)
			after := time.Now()
			if err != nil {
				t.Fatalf("GetSignedURL() failed: %v", err)
			}
			if want := "signed:bucket/" + object; got != want {
				t.Errorf("GetSignedURL() = %q, want %q", got, want)
			}
			calls := signer.recorded()
			if len(calls) != 1 {
				t.Fatalf("signer called %d times, want 1", len(calls))
			}
			c := calls[0]
			if c.bucket != "bucket" || c.object != object {
				t.Errorf("signer got (%q, %q), want (%q, %q)", c.bucket, c.object, "bucket", object)
			}
			if c.opts.Method != tc.wantMethod || c.opts.Scheme != tc.wantScheme {
				t.Errorf("signer got method %q and scheme %v, want %q and %v", c.opts.Method, c.opts.Scheme, tc.wantMethod, tc.wantScheme)
			}
			if c.opts.Expires.Before(before.Add(tc.wantExpiry)) || c.opts.Expires.After(after.Add(tc.wantExpiry)) {
				t.Errorf("signer got Expires %v, want %v after the call", c.opts.Expires, tc.wantExpiry)
			}
		})
	}

	t.Run("expiry from the context clock", func(t *testing.T) {
		signer := &recordingSigner{}
		ctx := platform.WithTimeProvider(t.Context(), func() time.Time {
			return time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
		})
		if _, err := svc.GetSignedURL(ctx, &GetSignedURLRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "user:profile.png",
			Expiration: 2 * time.Hour, Signer: signer.sign,
		}); err != nil {
			t.Fatalf("GetSignedURL() failed: %v", err)
		}
		calls := signer.recorded()
		if len(calls) != 1 {
			t.Fatalf("signer called %d times, want 1", len(calls))
		}
		if want := time.Date(2030, 1, 1, 2, 0, 0, 0, time.UTC); !calls[0].opts.Expires.Equal(want) {
			t.Errorf("signer got Expires %v, want %v", calls[0].opts.Expires, want)
		}
	})

	t.Run("signer error", func(t *testing.T) {
		wantErr := errors.New("key service unavailable")
		signer := &recordingSigner{err: wantErr}
		got, err := svc.GetSignedURL(t.Context(), &GetSignedURLRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "user:profile.png", Signer: signer.sign,
		})
		if got != "" || !errors.Is(err, wantErr) {
			t.Errorf("GetSignedURL() = (%q, %v), want (\"\", %v)", got, err, wantErr)
		}
	})

	for _, fileName := range []string{"none.txt", "ext.pdf"} {
		t.Run("no signing for "+fileName, func(t *testing.T) {
			signer := &recordingSigner{}
			got, err := svc.GetSignedURL(t.Context(), &GetSignedURLRequest{
				AppName: "app", UserID: "user1", SessionID: "sess1", FileName: fileName, Signer: signer.sign,
			})
			if got != "" || err != nil {
				t.Errorf("GetSignedURL() = (%q, %v), want (\"\", nil)", got, err)
			}
			if calls := signer.recorded(); len(calls) != 0 {
				t.Errorf("signer called %d times, want 0", len(calls))
			}
		})
	}
}

func TestURLProviderConcurrentUse(t *testing.T) {
	svc := newGCSServiceForTesting("bucket")
	mustSave(t, svc, "sess1", "f.txt", genai.NewPartFromText("v1"))
	signer := &recordingSigner{}
	req := &GetSignedURLRequest{AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "f.txt", Signer: signer.sign}

	const lookups = 8
	urls := make([]string, lookups)
	errs := make([]error, lookups+1)
	var wg sync.WaitGroup
	for i := range lookups {
		wg.Go(func() {
			urls[i], errs[i] = svc.GetSignedURL(t.Context(), req)
		})
	}
	wg.Go(func() {
		_, errs[lookups] = svc.Save(t.Context(), &artifact.SaveRequest{
			AppName: "app", UserID: "user1", SessionID: "sess1", FileName: "f.txt", Part: genai.NewPartFromText("v2"),
		})
	})
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("call %d failed: %v", i, err)
		}
	}
	for i, got := range urls {
		if got != "signed:bucket/app/user1/sess1/f.txt/1" && got != "signed:bucket/app/user1/sess1/f.txt/2" {
			t.Errorf("lookup %d = %q, want version 1 or 2 of f.txt", i, got)
		}
	}
	if req.Expiration != 0 || req.Method != "" {
		t.Errorf("request after the calls has Expiration %v and Method %q, want 0 and empty", req.Expiration, req.Method)
	}
}

func TestNewServiceReturnsURLProvider(t *testing.T) {
	svc, err := NewService(t.Context(), "bucket", option.WithoutAuthentication())
	if err != nil {
		t.Fatalf("NewService() failed: %v", err)
	}
	if _, ok := svc.(URLProvider); !ok {
		t.Errorf("NewService() returned %T, which does not implement URLProvider", svc)
	}
}
