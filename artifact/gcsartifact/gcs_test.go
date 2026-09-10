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
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"cloud.google.com/go/storage"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/iterator"
	"google.golang.org/genai"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/artifact/tests"
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
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := srv.Save(t.Context(), saveReq())
			if err != nil {
				errs[i] = err
				return
			}
			gotVersions[i] = resp.Version
		}()
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

	if _, err := svc.Save(t.Context(), saveReq()); !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("Save() err = %v, want ErrVersionConflict", err)
	}
	if got := fb.closeCalls; got != maxSaveAttempts {
		t.Errorf("write attempts = %d, want %d", got, maxSaveAttempts)
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
// already-cancelled context, and a normal short wait.
func TestSleepContext(t *testing.T) {
	if err := sleepContext(t.Context(), 0); err != nil {
		t.Errorf("sleepContext(_, 0) = %v, want nil", err)
	}

	cancelled, cancel := context.WithCancel(t.Context())
	cancel()
	if err := sleepContext(cancelled, time.Hour); !errors.Is(err, context.Canceled) {
		t.Errorf("sleepContext(cancelled, 1h) = %v, want context.Canceled", err)
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

	// Test hooks (guarded by mu): closeErr, when set, makes every writer Close
	// return it (a simulated write failure); closeCalls counts Close calls.
	closeErr   error
	closeCalls int

	// attrsErr, when set, is returned by every object's attrs() call in place
	// of the default not-found/found behavior. Used to simulate the GCS
	// client library's wrapped storage.ErrObjectNotExist (see
	// TestNotFoundIsWrappedSentinel).
	attrsErr error
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
	return &storage.ObjectAttrs{Name: b.name, Created: time.Now(), ContentType: b.contentType}, nil
}

// delete removes the object from the in-memory store.
func (o *fakeObject) delete(ctx context.Context) error {
	b := o.blob
	b.mu.Lock()
	defer b.mu.Unlock()
	b.exists = false
	b.data = nil
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
}

func (w *fakeWriter) Write(p []byte) (n int, err error) {
	return w.buffer.Write(p)
}

// Close commits the buffered data under the blob lock, so a conditional write
// that loses the race sees the winner's data and returns HTTP 412 like GCS. A
// bucket-level closeErr hook lets tests force a write failure.
func (w *fakeWriter) Close() error {
	if bkt := w.obj.bucket; bkt != nil {
		bkt.mu.Lock()
		bkt.closeCalls++
		forced := bkt.closeErr
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
	b.exists = true
	return nil
}

// SetContentType implements the final piece of the interface.
func (w *fakeWriter) SetContentType(cType string) {
	w.contentType = cType
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
