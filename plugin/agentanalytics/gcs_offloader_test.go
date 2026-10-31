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

package agentanalytics

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
	"unicode"
	"unicode/utf8"

	bq "cloud.google.com/go/bigquery"
	bqstorage "cloud.google.com/go/bigquery/storage/apiv1"
	"cloud.google.com/go/bigquery/storage/apiv1/storagepb"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	statuspb "google.golang.org/genproto/googleapis/rpc/status"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
)

// resolveUploads calls resolveUploadsForBatch with canDefer enabled for unit
// tests that exercise upload resolution directly.
func (b *BatchProcessor) resolveUploads(ctx context.Context, pauseOnCallerDeadline bool, batch []map[string]any) {
	b.resolveUploadsForBatch(ctx, uploadResolveOptions{
		pauseOnCallerDeadline: pauseOnCallerDeadline,
		canDefer:              true,
	}, batch)
}

// fakeStorage is a fake of the Cloud Storage JSON API. It handles the
// multipart object uploads that the storage client makes, and keeps the
// objects it was given.
type fakeStorage struct {
	srv *httptest.Server

	mu       sync.Mutex
	objects  map[string]storedObject // by "<bucket>/<name>"
	requests []url.Values            // the query of each request
	// failWith, if not zero, is the HTTP status of every upload.
	failWith int
}

type storedObject struct {
	contentType string
	data        []byte
}

func newFakeStorage(t *testing.T) *fakeStorage {
	t.Helper()
	// The storage client reads this variable, and it would override the
	// endpoint that the tests set.
	t.Setenv("STORAGE_EMULATOR_HOST", "")
	f := &fakeStorage{objects: map[string]storedObject{}}
	f.srv = httptest.NewServer(f)
	t.Cleanup(f.srv.Close)
	return f
}

// clientOptions returns the options that point a storage client at f.
func (f *fakeStorage) clientOptions() []option.ClientOption {
	return []option.ClientOption{
		option.WithEndpoint(f.srv.URL + "/storage/v1/"),
		option.WithoutAuthentication(),
	}
}

func (f *fakeStorage) object(bucket, name string) (storedObject, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	o, ok := f.objects[bucket+"/"+name]
	return o, ok
}

func (f *fakeStorage) queries() []url.Values {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.requests)
}

func (f *fakeStorage) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	bucket, ok := strings.CutPrefix(r.URL.Path, "/upload/storage/v1/b/")
	if !ok || r.Method != http.MethodPost {
		writeStorageError(w, http.StatusNotFound, "notFound")
		return
	}
	bucket = strings.TrimSuffix(bucket, "/o")
	query := r.URL.Query()
	f.mu.Lock()
	f.requests = append(f.requests, query)
	failWith := f.failWith
	f.mu.Unlock()
	if failWith != 0 {
		writeStorageError(w, failWith, "forbidden")
		return
	}

	mediaType, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || !strings.HasPrefix(mediaType, "multipart/") {
		writeStorageError(w, http.StatusBadRequest, "badRequest")
		return
	}
	parts := multipart.NewReader(r.Body, params["boundary"])
	metadataPart, err := parts.NextPart()
	if err != nil {
		writeStorageError(w, http.StatusBadRequest, "badRequest")
		return
	}
	var metadata struct {
		ContentType string `json:"contentType"`
	}
	if err := json.NewDecoder(metadataPart).Decode(&metadata); err != nil {
		writeStorageError(w, http.StatusBadRequest, "badRequest")
		return
	}
	mediaPart, err := parts.NextPart()
	if err != nil {
		writeStorageError(w, http.StatusBadRequest, "badRequest")
		return
	}
	data, err := io.ReadAll(mediaPart)
	if err != nil {
		writeStorageError(w, http.StatusBadRequest, "badRequest")
		return
	}

	name := query.Get("name")
	f.mu.Lock()
	_, exists := f.objects[bucket+"/"+name]
	if query.Get("ifGenerationMatch") == "0" && exists {
		f.mu.Unlock()
		writeStorageError(w, http.StatusPreconditionFailed, "conditionNotMet")
		return
	}
	f.objects[bucket+"/"+name] = storedObject{contentType: metadata.ContentType, data: data}
	f.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"kind": "storage#object", "bucket": bucket, "name": name, "generation": "1",
		"size": strconv.Itoa(len(data)), "contentType": metadata.ContentType,
	})
}

func writeStorageError(w http.ResponseWriter, code int, reason string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_, _ = fmt.Fprintf(w, `{"error":{"code":%d,"message":%q,"errors":[{"message":%q,"domain":"global","reason":%q}]}}`,
		code, reason, reason, reason)
}

// recordingTransport is an http.RoundTripper that records, in order, the
// requests it carries and the calls to CloseIdleConnections, which is what
// closing a storage client does.
type recordingTransport struct {
	inner http.RoundTripper

	mu     sync.Mutex
	events []string
}

func (t *recordingTransport) record(event string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.events = append(t.events, event)
}

func (t *recordingTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	t.record("request " + r.Method)
	return t.inner.RoundTrip(r)
}

func (t *recordingTransport) CloseIdleConnections() {
	t.record("close")
	if c, ok := t.inner.(interface{ CloseIdleConnections() }); ok {
		c.CloseIdleConnections()
	}
}

func (t *recordingTransport) recorded() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return slices.Clone(t.events)
}

func TestGCSOffloader_UploadsToCloudStorage(t *testing.T) {
	ctx := context.Background()
	fs := newFakeStorage(t)
	o, err := newGCSOffloader(ctx, "test-bucket", fs.clientOptions()...)
	if err != nil {
		t.Fatalf("newGCSOffloader() error: %v", err)
	}
	t.Cleanup(func() { _ = o.Close() })

	data := []byte{0x89, 'P', 'N', 'G', 0, 1, 2}
	uri, err := o.UploadContent(ctx, GCSUploadRequest{Path: "logs/2026-03-04/t/s_e_c0_p0.png", Data: data, ContentType: "image/png"})
	if err != nil {
		t.Fatalf("UploadContent() error: %v", err)
	}
	if want := "gs://test-bucket/logs/2026-03-04/t/s_e_c0_p0.png"; uri != want {
		t.Errorf("UploadContent() = %q, want %q", uri, want)
	}
	got, ok := fs.object("test-bucket", "logs/2026-03-04/t/s_e_c0_p0.png")
	if !ok {
		t.Fatalf("the object was not stored; got %v", fs.objects)
	}
	if got.contentType != "image/png" || !bytes.Equal(got.data, data) {
		t.Errorf("stored object = {%q, %x}, want {image/png, %x}", got.contentType, got.data, data)
	}
	// The write must be conditional, so that it never overwrites an object.
	queries := fs.queries()
	if len(queries) != 1 {
		t.Fatalf("got %d requests, want 1", len(queries))
	}
	if got := queries[0].Get("ifGenerationMatch"); got != "0" {
		t.Errorf("ifGenerationMatch = %q, want \"0\" (create only if the object does not exist)", got)
	}
}

func TestGCSOffloader_ExistingObjectCountsAsUploaded(t *testing.T) {
	ctx := context.Background()
	fs := newFakeStorage(t)
	o, err := newGCSOffloader(ctx, "test-bucket", fs.clientOptions()...)
	if err != nil {
		t.Fatalf("newGCSOffloader() error: %v", err)
	}
	t.Cleanup(func() { _ = o.Close() })

	req := GCSUploadRequest{Path: "same/object.txt", Data: []byte("first"), ContentType: "text/plain"}
	if _, err := o.UploadContent(ctx, req); err != nil {
		t.Fatalf("first UploadContent() error: %v", err)
	}
	// A retried write of the same part finds the object of the first attempt.
	req.Data = []byte("second")
	uri, err := o.UploadContent(ctx, req)
	if err != nil {
		t.Fatalf("second UploadContent() error: %v, want the 412 to count as success", err)
	}
	if want := "gs://test-bucket/same/object.txt"; uri != want {
		t.Errorf("second UploadContent() = %q, want %q", uri, want)
	}
	if got, _ := fs.object("test-bucket", "same/object.txt"); string(got.data) != "first" {
		t.Errorf("stored data = %q, want the first write %q to be kept", got.data, "first")
	}
}

func TestGCSOffloader_UploadErrors(t *testing.T) {
	for _, status := range []int{http.StatusForbidden, http.StatusNotFound, http.StatusBadRequest} {
		t.Run(strconv.Itoa(status), func(t *testing.T) {
			ctx := context.Background()
			fs := newFakeStorage(t)
			fs.failWith = status
			o, err := newGCSOffloader(ctx, "test-bucket", fs.clientOptions()...)
			if err != nil {
				t.Fatalf("newGCSOffloader() error: %v", err)
			}
			t.Cleanup(func() { _ = o.Close() })

			uri, err := o.UploadContent(ctx, GCSUploadRequest{Path: "p.txt", Data: []byte("PAYLOAD-MARKER"), ContentType: "text/plain"})
			if err == nil {
				t.Fatalf("UploadContent() = %q, nil; want an error", uri)
			}
			if uri != "" {
				t.Errorf("UploadContent() uri = %q, want empty on error", uri)
			}
			var apiErr *googleapi.Error
			if !errors.As(err, &apiErr) || apiErr.Code != status {
				t.Errorf("UploadContent() error = %v, want a *googleapi.Error with code %d", err, status)
			}
			if strings.Contains(err.Error(), "PAYLOAD-MARKER") {
				t.Errorf("the error contains the payload: %v", err)
			}
		})
	}
}

func TestGCSOffloader_CanceledContext(t *testing.T) {
	fs := newFakeStorage(t)
	o, err := newGCSOffloader(context.Background(), "test-bucket", fs.clientOptions()...)
	if err != nil {
		t.Fatalf("newGCSOffloader() error: %v", err)
	}
	t.Cleanup(func() { _ = o.Close() })

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if uri, err := o.UploadContent(ctx, GCSUploadRequest{Path: "p.txt", Data: []byte("x"), ContentType: "text/plain"}); err == nil {
		t.Errorf("UploadContent() = %q, nil with a canceled context; want an error", uri)
	}
	if _, ok := fs.object("test-bucket", "p.txt"); ok {
		t.Error("the object was stored although the context was canceled")
	}
}

func TestIsPreconditionFailed(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "412", err: &googleapi.Error{Code: http.StatusPreconditionFailed}, want: true},
		{name: "wrapped 412", err: fmt.Errorf("write: %w", &googleapi.Error{Code: http.StatusPreconditionFailed}), want: true},
		{name: "403", err: &googleapi.Error{Code: http.StatusForbidden}, want: false},
		{name: "404", err: &googleapi.Error{Code: http.StatusNotFound}, want: false},
		{name: "other error", err: errors.New("412"), want: false},
		{name: "nil", err: nil, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isPreconditionFailed(tt.err); got != tt.want {
				t.Errorf("isPreconditionFailed(%v) = %v, want %v", tt.err, got, tt.want)
			}
		})
	}
}

func TestGCSObjectPath(t *testing.T) {
	ts := time.Date(2026, 3, 4, 5, 6, 7, 0, time.UTC)
	eastern := time.FixedZone("UTC-5", -5*60*60)
	tests := []struct {
		name                     string
		prefix                   string
		ts                       time.Time
		traceID, spanID, eventID string
		contentOrd, partIdx      int
		ext                      string
		want                     string
	}{
		{
			name: "all parts", ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			contentOrd: 1, partIdx: 2, ext: ".png",
			want: "2026-03-04/trace/span_event_c1_p2.png",
		},
		{
			name: "prefix", prefix: "agent-logs", ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: "agent-logs/2026-03-04/trace/span_event_c0_p0.txt",
		},
		{
			name: "prefix with slashes", prefix: "/a/b/", ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: "a/b/2026-03-04/trace/span_event_c0_p0.txt",
		},
		{
			name: "only slashes in the prefix", prefix: "//", ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: "2026-03-04/trace/span_event_c0_p0.txt",
		},
		{
			name: "the date is in UTC", ts: time.Date(2026, 3, 4, 23, 30, 0, 0, eastern), traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: "2026-03-05/trace/span_event_c0_p0.txt",
		},
		{
			name: "missing ids", ts: ts, ext: ".bin",
			want: "2026-03-04/no_trace/no_span_no_event_c0_p0.bin",
		},
		{
			name: "multi-digit ordinals", ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			contentOrd: 12, partIdx: 345, ext: ".jpg",
			want: "2026-03-04/trace/span_event_c12_p345.jpg",
		},
		{
			// Unsafe runes are replaced with underscores and disambiguated with
			// a deterministic SHA-256 suffix of the raw identifier.
			name: "unsafe characters are replaced", ts: ts, traceID: "a/b c", spanID: "x\ny", eventID: "é?*",
			ext: ".txt",
			want: fmt.Sprintf("2026-03-04/a_b_c%s/x_y%s____%s_c0_p0.txt",
				pathSegmentHashSuffix("a/b c"),
				pathSegmentHashSuffix("x\ny"),
				pathSegmentHashSuffix("é?*")),
		},
		{
			name:   "dot-relative and empty inner prefix segments are preserved",
			prefix: "../a/../b/./c//", ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: "../a/../b/./c/2026-03-04/trace/span_event_c0_p0.txt",
		},
		{
			name:   "dot-only prefix is preserved after slash trim",
			prefix: "./..", ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: "./../2026-03-04/trace/span_event_c0_p0.txt",
		},
		{
			name:   "backslashes in prefix are preserved",
			prefix: `team\prod/v1`, ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: `team\prod/v1/2026-03-04/trace/span_event_c0_p0.txt`,
		},
		{
			name:   "tildes in prefix are rewritten with hash suffix",
			prefix: `~team\prod/v~1`, ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: `_team\prod/v_1` + pathSegmentHashSuffix(`~team\prod/v~1`) + `/2026-03-04/trace/span_event_c0_p0.txt`,
		},
		{
			name:   "non-matching .well-known prefix is preserved",
			prefix: ".well-known/acme-challenge-x", ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: ".well-known/acme-challenge-x/2026-03-04/trace/span_event_c0_p0.txt",
		},
		{
			name:   "triple-dot prefix segment is preserved",
			prefix: "a/.../b", ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: "a/.../b/2026-03-04/trace/span_event_c0_p0.txt",
		},
		{
			name:   "valid prefixes with spaces, equals, and unicode are preserved",
			prefix: "tenant=acme/logs v1/テナント", ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: "tenant=acme/logs v1/テナント/2026-03-04/trace/span_event_c0_p0.txt",
		},
		{
			name:   "reserved acme-challenge prefix is rewritten",
			prefix: ".well-known/acme-challenge/x", ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: fmt.Sprintf("_well-known/acme-challenge/x%s/2026-03-04/trace/span_event_c0_p0.txt", pathSegmentHashSuffix(".well-known/acme-challenge/x")),
		},
		{
			name:   "control characters in prefix are rewritten",
			prefix: "logs\nx", ts: ts, traceID: "trace", spanID: "span", eventID: "event",
			ext:  ".txt",
			want: fmt.Sprintf("logs_x%s/2026-03-04/trace/span_event_c0_p0.txt", pathSegmentHashSuffix("logs\nx")),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := gcsObjectPath(tt.prefix, tt.ts, tt.traceID, tt.spanID, tt.eventID, tt.contentOrd, tt.partIdx, tt.ext); got != tt.want {
				t.Errorf("gcsObjectPath() = %q, want %q", got, tt.want)
			}
		})
	}
}

// uploadAttempt is one call of fakeOffloader.UploadContent.
type uploadAttempt struct {
	Path        string
	ContentType string
	// Data is a copy of the data of the request, made after the hook ran.
	Data []byte
	// HasDeadline and Remaining describe the context of the call.
	HasDeadline bool
	Remaining   time.Duration
}

// fakeOffloader is a GCSContentOffloader that records the calls it gets.
type fakeOffloader struct {
	// hook, if set, runs at the start of every call. The call fails with the
	// error that it returns.
	hook func(ctx context.Context, req GCSUploadRequest) error
	// uri, if set, makes the URI that a call returns. By default that is
	// "gs://fake-bucket/" followed by the path.
	uri func(req GCSUploadRequest) string

	mu          sync.Mutex
	calls       []uploadAttempt
	closes      int
	inFlight    int
	maxInFlight int
}

var _ GCSContentOffloader = (*fakeOffloader)(nil)

func (f *fakeOffloader) UploadContent(ctx context.Context, req GCSUploadRequest) (string, error) {
	f.mu.Lock()
	f.inFlight++
	f.maxInFlight = max(f.maxInFlight, f.inFlight)
	f.mu.Unlock()

	call := uploadAttempt{Path: req.Path, ContentType: req.ContentType}
	if deadline, ok := ctx.Deadline(); ok {
		call.HasDeadline, call.Remaining = true, time.Until(deadline)
	}
	// The call is recorded in a defer, so that a call that panics is too.
	defer func() {
		f.mu.Lock()
		defer f.mu.Unlock()
		f.inFlight--
		f.calls = append(f.calls, call)
	}()
	if f.hook != nil {
		if err := f.hook(ctx, req); err != nil {
			return "", err
		}
	}
	// The data is copied after the hook, which may wait for a while: what
	// counts is what the offloader sees when it gets to use the data.
	call.Data = bytes.Clone(req.Data)
	if f.uri != nil {
		return f.uri(req), nil
	}
	return "gs://fake-bucket/" + req.Path, nil
}

func (f *fakeOffloader) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closes++
	return nil
}

// uploads returns the calls so far, ordered by path.
func (f *fakeOffloader) uploads() []uploadAttempt {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := slices.Clone(f.calls)
	slices.SortFunc(out, func(a, b uploadAttempt) int { return strings.Compare(a.Path, b.Path) })
	return out
}

// byPath returns the calls so far by their paths.
func (f *fakeOffloader) byPath() map[string]uploadAttempt {
	out := map[string]uploadAttempt{}
	for _, c := range f.uploads() {
		out[c.Path] = c
	}
	return out
}

func (f *fakeOffloader) closeCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.closes
}

// maxConcurrent returns the most calls that were in progress at once.
func (f *fakeOffloader) maxConcurrent() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.maxInFlight
}

// imageContent returns a message with one PNG part that holds data.
func imageContent(data ...byte) *genai.Content {
	return &genai.Content{Role: "user", Parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "image/png", Data: data}}}}
}

// imageContents returns n messages with one PNG part each, so that a row
// queued for them has n uploads.
func imageContents(n int) []*genai.Content {
	contents := make([]*genai.Content, n)
	for i := range contents {
		contents[i] = imageContent(1, 2, 3)
	}
	return contents
}

// queuedRow returns the row that the plugin queues for contents, which are
// numbered from 0 as the messages of a request are, with the ids that the
// names of objects are made of.
func queuedRow(opts partsOptions, contents ...*genai.Content) map[string]any {
	c := &partsCollector{opts: opts}
	for i, content := range contents {
		c.addContent(content, i)
	}
	row := map[string]any{
		timestampField:     time.Date(2026, 3, 4, 5, 6, 7, 0, time.UTC),
		eventIDField:       "event1",
		"trace_id":         "trace1",
		"span_id":          "span1",
		columnContentParts: c.parts,
	}
	if len(c.pending) > 0 {
		row[rawPartsKey] = c.pending
	}
	return row
}

// barrier releases the callers of wait once n of them have arrived, or fails
// them if that takes too long.
type barrier struct {
	n       int32
	arrived atomic.Int32
	open    chan struct{}
}

func newBarrier(n int) *barrier {
	return &barrier{n: int32(n), open: make(chan struct{})}
}

func (b *barrier) wait() error {
	if b.arrived.Add(1) == b.n {
		close(b.open)
	}
	select {
	case <-b.open:
		return nil
	case <-time.After(5 * time.Second):
		return fmt.Errorf("fewer than %d callers arrived", b.n)
	}
}

func TestResolveUploads_RewritesTheEntries(t *testing.T) {
	text := "SECRET-TEXT-MARKER" + strings.Repeat("-", 100)
	row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 50},
		&genai.Content{Role: "user", Parts: []*genai.Part{
			{Text: text},
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}}},
		}})
	off := &fakeOffloader{}
	bp := &BatchProcessor{config: Config{GCSOffloader: off, GCSPathPrefix: "logs", ConnectionID: "us.conn"}}

	bp.resolveUploads(context.Background(), false, []map[string]any{row})

	textURI := "gs://fake-bucket/logs/2026-03-04/trace1/span1_event1_c0_p0.txt"
	imageURI := "gs://fake-bucket/logs/2026-03-04/trace1/span1_event1_c0_p1.png"
	want := []map[string]any{
		{
			"part_index": 0, "storage_mode": "GCS_REFERENCE", "mime_type": "text/plain",
			"text": text + "... [OFFLOADED]", "uri": textURI,
			"object_ref": map[string]any{
				"uri": textURI, "authorizer": "us.conn",
				"details": `{"gcs_metadata":{"content_type":"text/plain; charset=utf-8","size":118}}`,
			},
		},
		{
			"part_index": 1, "storage_mode": "GCS_REFERENCE", "mime_type": "image/png",
			"text": "[MEDIA OFFLOADED]", "uri": imageURI,
			"object_ref": map[string]any{
				"uri": imageURI, "authorizer": "us.conn",
				"details": `{"gcs_metadata":{"content_type":"image/png","size":3}}`,
			},
		},
	}
	if diff := cmp.Diff(want, row[columnContentParts]); diff != "" {
		t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
	}
	if _, ok := row[rawPartsKey]; ok {
		t.Errorf("the row still has %q; it is not a column", rawPartsKey)
	}
	// On a successful resolution with no deferred uploads left, resolveUploads
	// cleans up the deferredUploadsKey from the row before the caller settles
	// the batch (releaseBatchUploads / clearDeferredUploads).
	if _, ok := row[deferredUploadsKey]; ok {
		t.Errorf("the row still has %q after resolveUploads; empty deferred state must be cleaned up", deferredUploadsKey)
	}
	wantCalls := []uploadAttempt{
		{Path: "logs/2026-03-04/trace1/span1_event1_c0_p0.txt", ContentType: "text/plain; charset=utf-8", Data: []byte(text)},
		{Path: "logs/2026-03-04/trace1/span1_event1_c0_p1.png", ContentType: "image/png", Data: []byte{1, 2, 3}},
	}
	got := off.uploads()
	if diff := cmp.Diff(wantCalls, got, cmpopts.IgnoreFields(uploadAttempt{}, "HasDeadline", "Remaining")); diff != "" {
		t.Errorf("uploads mismatch (-want +got):\n%s", diff)
	}
}

func TestResolveUploads_Authorizer(t *testing.T) {
	tests := []struct {
		name         string
		connectionID string
		want         map[string]any
	}{
		{
			name:         "without a connection there is no authorizer",
			connectionID: "",
			want: map[string]any{
				"uri": "gs://fake-bucket/2026-03-04/trace1/span1_event1_c0_p0.png", "details": `{"gcs_metadata":{"content_type":"image/png","size":1}}`,
			},
		},
		{
			name:         "the connection is the authorizer",
			connectionID: "us.conn",
			want: map[string]any{
				"uri": "gs://fake-bucket/2026-03-04/trace1/span1_event1_c0_p0.png", "details": `{"gcs_metadata":{"content_type":"image/png","size":1}}`,
				"authorizer": "us.conn",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}, imageContent(7))
			bp := &BatchProcessor{config: Config{GCSOffloader: &fakeOffloader{}, ConnectionID: tt.connectionID}}

			bp.resolveUploads(context.Background(), false, []map[string]any{row})

			ref := row[columnContentParts].([]map[string]any)[0]["object_ref"]
			// version is left out in both cases: adk-python writes NULL, and
			// an empty string would be an invalid object version.
			if diff := cmp.Diff(tt.want, ref); diff != "" {
				t.Errorf("object_ref mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestResolveUploads_FailuresKeepTheInlineForm checks that every way an upload
// can go wrong leaves text entries as they were queued and marks binary
// entries with [UPLOAD FAILED], and that none of them logs what was to be
// uploaded.
func TestResolveUploads_FailuresKeepTheInlineForm(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	tests := []struct {
		name string
		ctx  context.Context
		// timeout overrides the timeout of an upload.
		timeout time.Duration
		hook    func(ctx context.Context, req GCSUploadRequest) error
		uri     func(req GCSUploadRequest) string
		// wantLog is what the log has to say; wantCalls how many uploads
		// are started.
		wantLog   string
		wantCalls int
	}{
		{
			name: "the offloader fails",
			hook: func(context.Context, GCSUploadRequest) error {
				return errors.New("storage unavailable: SECRET-ERR-MARKER")
			},
			wantLog:   "error type *errors.errorString",
			wantCalls: 2,
		},
		{
			name:      "the offloader panics",
			hook:      func(context.Context, GCSUploadRequest) error { panic("PANIC-MARKER") },
			wantLog:   "offloader panicked (string)",
			wantCalls: 2,
		},
		{
			name:      "the offloader returns no URI",
			uri:       func(GCSUploadRequest) string { return "" },
			wantLog:   "not a gs:// URI",
			wantCalls: 2,
		},
		{
			name:      "the offloader returns a non-gs URI",
			uri:       func(GCSUploadRequest) string { return "https://example.com/SECRET-URL-MARKER" },
			wantLog:   "not a gs:// URI",
			wantCalls: 2,
		},
		{
			name:    "the upload takes too long",
			timeout: 20 * time.Millisecond,
			hook: func(ctx context.Context, _ GCSUploadRequest) error {
				<-ctx.Done()
				return ctx.Err()
			},
			wantLog:   "deadline exceeded",
			wantCalls: 2,
		},
		{
			name:      "the context is canceled",
			ctx:       canceled,
			wantLog:   "context canceled",
			wantCalls: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			content := &genai.Content{Role: "user", Parts: []*genai.Part{
				{Text: "SECRET-TEXT-MARKER" + strings.Repeat("-", 100)},
				{InlineData: &genai.Blob{MIMEType: `image/png; name="SECRET-MIME-PARAM"`, Data: []byte("SECRET-IMAGE-MARKER")}},
			}}
			opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}
			row := queuedRow(opts, content)
			// Text keeps its inline cut; binary parts mark the failed upload.
			opts.offload = false
			want := queuedRow(opts, content)[columnContentParts].([]map[string]any)
			want[1]["text"] = uploadFailedText
			opts.offload = true
			off := &fakeOffloader{hook: tt.hook, uri: tt.uri}
			logger := &recordingLogger{}
			bp := &BatchProcessor{config: Config{GCSOffloader: off, Logger: logger}, uploadTimeout: tt.timeout}
			ctx := tt.ctx
			if ctx == nil {
				ctx = context.Background()
			}

			bp.resolveUploads(ctx, false, []map[string]any{row})
			// This test never requeues. Drop a deferred upload the way a
			// successful write does; that is what logs it. Other failures
			// were already logged.
			bp.clearDeferredUploads([]map[string]any{row})

			if diff := cmp.Diff(want, row[columnContentParts]); diff != "" {
				t.Errorf("content_parts changed (-want +got):\n%s", diff)
			}
			if _, ok := row[rawPartsKey]; ok {
				t.Errorf("the row still has %q; it is not a column", rawPartsKey)
			}
			if got := len(off.uploads()); got != tt.wantCalls {
				t.Errorf("%d uploads were started, want %d", got, tt.wantCalls)
			}
			if !logger.contains("GCS upload failed") || !logger.contains(tt.wantLog) {
				t.Errorf("the log lacks %q and \"GCS upload failed\":\n%s", tt.wantLog, logger)
			}
			for _, secret := range []string{"SECRET-TEXT-MARKER", "SECRET-IMAGE-MARKER", "PANIC-MARKER", "SECRET-ERR-MARKER", "SECRET-URL-MARKER", "SECRET-MIME-PARAM"} {
				if logger.contains(secret) {
					t.Errorf("the log contains %q:\n%s", secret, logger)
				}
			}
		})
	}
}

func TestResolveUploads_OneFailureDoesNotSpoilTheOthers(t *testing.T) {
	row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}, imageContent(1), imageContent(2), imageContent(3))
	off := &fakeOffloader{hook: func(_ context.Context, req GCSUploadRequest) error {
		if strings.Contains(req.Path, "_c1_") {
			return errors.New("boom")
		}
		return nil
	}}
	bp := &BatchProcessor{config: Config{GCSOffloader: off, Logger: &recordingLogger{}}}

	bp.resolveUploads(context.Background(), false, []map[string]any{row})

	var got []string
	for _, part := range row[columnContentParts].([]map[string]any) {
		got = append(got, part["storage_mode"].(string))
	}
	if diff := cmp.Diff([]string{"GCS_REFERENCE", "INLINE", "GCS_REFERENCE"}, got); diff != "" {
		t.Errorf("storage modes mismatch (-want +got):\n%s", diff)
	}
}

func TestResolveUploads_UploadsHaveADeadline(t *testing.T) {
	row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}, imageContent(1))
	off := &fakeOffloader{}
	bp := &BatchProcessor{config: Config{GCSOffloader: off}}

	bp.resolveUploads(context.Background(), false, []map[string]any{row})

	calls := off.uploads()
	if len(calls) != 1 {
		t.Fatalf("got %d uploads, want 1", len(calls))
	}
	// An upload may take 10 seconds, no more and, as it starts at once, not
	// much less.
	const want = 10 * time.Second
	if got := calls[0]; !got.HasDeadline || got.Remaining <= want-5*time.Second || got.Remaining > want {
		t.Errorf("the upload context has deadline %v in %v; want one in (%v, %v]", got.HasDeadline, got.Remaining, want-5*time.Second, want)
	}
}

func TestResolveUploads_ConcurrencyIsBounded(t *testing.T) {
	tests := []struct {
		name  string
		limit int
		want  int
	}{
		{name: "the configured limit", limit: 3, want: 3},
		{name: "a limit of zero means the default", limit: 0, want: defaultGCSMaxConcurrentUploads},
		{name: "a negative limit means the default", limit: -1, want: defaultGCSMaxConcurrentUploads},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			contents := make([]*genai.Content, 25)
			for i := range contents {
				contents[i] = imageContent(byte(i))
			}
			row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}, contents...)
			// The first tt.want uploads wait for each other, which shows that
			// that many run at once; the pause that follows gives more a
			// chance to start if the limit does not hold.
			gate := newBarrier(tt.want)
			off := &fakeOffloader{hook: func(context.Context, GCSUploadRequest) error {
				if err := gate.wait(); err != nil {
					return err
				}
				time.Sleep(10 * time.Millisecond)
				return nil
			}}
			bp := &BatchProcessor{config: Config{GCSOffloader: off, GCSMaxConcurrentUploads: tt.limit, Logger: &recordingLogger{}}}

			bp.resolveUploads(context.Background(), false, []map[string]any{row})

			if got := off.maxConcurrent(); got != tt.want {
				t.Errorf("%d uploads ran at the same time, want %d", got, tt.want)
			}
			if got := len(off.uploads()); got != len(contents) {
				t.Errorf("%d uploads were made, want %d", got, len(contents))
			}
			for i, part := range row[columnContentParts].([]map[string]any) {
				if part["storage_mode"] != "GCS_REFERENCE" {
					t.Errorf("part %d has storage mode %v, want GCS_REFERENCE", i, part["storage_mode"])
				}
			}
		})
	}
}

func TestResolveUploads_WithoutAnOffloaderThePartsStayInline(t *testing.T) {
	row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}, imageContent(1))
	opts := partsOptions{maxLen: 1000}
	want := queuedRow(opts, imageContent(1))[columnContentParts]
	logger := &recordingLogger{}
	bp := &BatchProcessor{config: Config{Logger: logger}}

	bp.resolveUploads(context.Background(), false, []map[string]any{row})

	if diff := cmp.Diff(want, row[columnContentParts]); diff != "" {
		t.Errorf("content_parts changed (-want +got):\n%s", diff)
	}
	if _, ok := row[rawPartsKey]; ok {
		t.Errorf("the row still has %q; it is not a column", rawPartsKey)
	}
	// On a successful resolution with no deferred uploads left, resolveUploads
	// cleans up the deferredUploadsKey from the row before the caller settles
	// the batch (releaseBatchUploads / clearDeferredUploads).
	if _, ok := row[deferredUploadsKey]; ok {
		t.Errorf("the row still has %q after resolveUploads; empty deferred state must be cleaned up", deferredUploadsKey)
	}
	// No upload is attempted, so none fails.
	if got := logger.String(); got != "" {
		t.Errorf("the log is not empty:\n%s", got)
	}
}

func TestResolveUploads_RowsWithoutUploadsAreLeftAlone(t *testing.T) {
	plain := map[string]any{"event_type": "X", columnContentParts: []map[string]any{{"part_index": 0}}}
	// Pending uploads that name an entry which is not there are skipped.
	outOfRange := map[string]any{
		columnContentParts: []map[string]any{{"part_index": 0}},
		rawPartsKey:        []pendingUpload{{slot: 1, mimeType: "image/png", data: []byte{1}}, {slot: -1, mimeType: "image/png", data: []byte{1}}},
	}
	noParts := map[string]any{rawPartsKey: []pendingUpload{{slot: 0, mimeType: "image/png", data: []byte{1}}}}
	off := &fakeOffloader{}
	bp := &BatchProcessor{config: Config{GCSOffloader: off}}

	bp.resolveUploads(context.Background(), false, []map[string]any{plain, outOfRange, noParts})

	if got := off.uploads(); len(got) != 0 {
		t.Errorf("got uploads %v, want none", got)
	}
	if diff := cmp.Diff([]map[string]any{{"part_index": 0}}, plain[columnContentParts]); diff != "" {
		t.Errorf("content_parts changed (-want +got):\n%s", diff)
	}
	for i, row := range []map[string]any{outOfRange, noParts} {
		if _, ok := row[rawPartsKey]; ok {
			t.Errorf("row %d still has %q", i, rawPartsKey)
		}
	}
}

func TestBatchProcessor_ObjectPath(t *testing.T) {
	bp := &BatchProcessor{config: Config{GCSPathPrefix: "p"}}
	u := pendingUpload{contentOrd: 2, partIdx: 3, mimeType: "image/jpeg"}

	t.Run("from the row", func(t *testing.T) {
		row := map[string]any{
			timestampField: time.Date(2026, 3, 4, 23, 30, 0, 0, time.FixedZone("UTC-5", -5*60*60)),
			eventIDField:   "e", "trace_id": "t", "span_id": "s",
		}
		if got, want := bp.objectPath(row, u), "p/2026-03-05/t/s_e_c2_p3.jpg"; got != want {
			t.Errorf("objectPath() = %q, want %q", got, want)
		}
	})
	t.Run("a row without ids or a time", func(t *testing.T) {
		before := time.Now().UTC().Format(time.DateOnly)
		got := bp.objectPath(map[string]any{}, u)
		after := time.Now().UTC().Format(time.DateOnly)
		rest := "/no_trace/no_span_no_event_c2_p3.jpg"
		if got != "p/"+before+rest && got != "p/"+after+rest {
			t.Errorf("objectPath() = %q, want today's date between p/ and %s", got, rest)
		}
	})
}

// withSpan returns ctx with a valid span context, from which a row takes its
// trace_id and span_id.
func withSpan(ctx context.Context, traceID, spanID string) context.Context {
	tid, _ := trace.TraceIDFromHex(traceID)
	sid, _ := trace.SpanIDFromHex(spanID)
	return trace.ContextWithSpanContext(ctx, trace.NewSpanContext(trace.SpanContextConfig{TraceID: tid, SpanID: sid}))
}

// offloadingPlugin is setupTestPluginWithConfig with an offloader and quick
// background flushes, so that flushes overlap with the callbacks.
func offloadingPlugin(t *testing.T, off GCSContentOffloader, mutate func(*Config)) (*baseplugin.Plugin, chan *storagepb.AppendRowsRequest, *mockInvocationContext) {
	t.Helper()
	return setupTestPluginWithConfig(t, func(c *Config) {
		c.GCSOffloader = off
		c.BatchFlushIntv = 10 * time.Millisecond
		if mutate != nil {
			mutate(c)
		}
	})
}

// logRequest has the plugin log req and returns the LLM_REQUEST row that is
// written once the invocation ends.
func logRequest(t *testing.T, p *baseplugin.Plugin, requests <-chan *storagepb.AppendRowsRequest, mCtx *mockInvocationContext, req *model.LLMRequest) map[string]any {
	t.Helper()
	if _, err := p.BeforeModelCallback()(mCtx, req); err != nil {
		t.Fatalf("BeforeModelCallback() error: %v", err)
	}
	p.AfterRunCallback()(mCtx)
	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	return onlyRowOfType(t, rows, EventLLMRequest)
}

// contentPartsOf returns the content_parts of a written row.
func contentPartsOf(t *testing.T, row map[string]any) []map[string]any {
	t.Helper()
	parts, ok := row[columnContentParts].([]map[string]any)
	if !ok {
		t.Fatalf("the %v row has no content_parts", row["event_type"])
	}
	return parts
}

// rowDate returns the UTC date of the timestamp of a written row, which is
// the date in the names of its objects.
func rowDate(row map[string]any) string {
	ts, _ := row[timestampField].(time.Time)
	return ts.UTC().Format(time.DateOnly)
}

// inlineImagePart returns the content_parts entry of an image that stays
// inline, as decoded from Arrow.
func inlineImagePart(index int64) map[string]any {
	return map[string]any{"mime_type": "image/png", "text": "[BINARY DATA]", "part_index": index, "storage_mode": "INLINE"}
}

// ignoreDeadlines makes cmp.Diff leave out how long a call had to run.
var ignoreDeadlines = cmpopts.IgnoreFields(uploadAttempt{}, "HasDeadline", "Remaining")

func TestGCSOffloading_HybridInlineAndGCSReference(t *testing.T) {
	off := &fakeOffloader{}
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
		c.GCSPathPrefix = "logs/"
		c.ConnectionID = "us.my-connection"
		c.MaxInlineTextSize = 40
	})
	const traceID, spanID = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
	mCtx.ctx = withSpan(mCtx.ctx, traceID, spanID)
	longText := strings.Repeat("0123456789", 10)
	png := []byte{0x89, 'P', 'N', 'G', 1, 2, 3}
	req := &model.LLMRequest{
		Model: "gemini-test",
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "short"}, {Text: longText}}},
			{Role: "user", Parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: png}},
				{FileData: &genai.FileData{FileURI: "gs://bucket/existing.pdf", MIMEType: "application/pdf"}},
				{FileData: &genai.FileData{FileURI: "https://example.com/doc.pdf", MIMEType: "application/pdf"}},
			}},
		},
		Config: &genai.GenerateContentConfig{SystemInstruction: textContent("", "be brief")},
	}

	row := logRequest(t, p, requests, mCtx, req)

	if row["trace_id"] != traceID || fmt.Sprint(row["span_id"]) == "" {
		t.Fatalf("row trace_id=%v span_id=%v, want trace_id=%s and non-empty span_id", row["trace_id"], row["span_id"], traceID)
	}
	base := fmt.Sprintf("logs/%s/%s/%v_%v", rowDate(row), traceID, row["span_id"], row[eventIDField])
	textObject, imageObject := base+"_c0_p1.txt", base+"_c1_p0.png"
	textURI, imageURI := "gs://fake-bucket/"+textObject, "gs://fake-bucket/"+imageObject
	want := []map[string]any{
		textPart("short", 0),
		{
			"part_index": int64(1), "storage_mode": "GCS_REFERENCE", "mime_type": "text/plain",
			"text": longText + "... [OFFLOADED]", "uri": textURI,
			"object_ref": map[string]any{
				"uri": textURI, "authorizer": "us.my-connection",
				"details": `{"gcs_metadata":{"content_type":"text/plain; charset=utf-8","size":100}}`,
			},
		},
		{
			"part_index": int64(0), "storage_mode": "GCS_REFERENCE", "mime_type": "image/png",
			"text": "[MEDIA OFFLOADED]", "uri": imageURI,
			"object_ref": map[string]any{
				"uri": imageURI, "authorizer": "us.my-connection",
				"details": `{"gcs_metadata":{"content_type":"image/png","size":7}}`,
			},
		},
		{
			"part_index": int64(1), "storage_mode": "EXTERNAL_URI", "mime_type": "application/pdf",
			"uri": "gs://bucket/existing.pdf",
			"object_ref": map[string]any{
				"uri": "gs://bucket/existing.pdf", "authorizer": "us.my-connection", "details": "{}",
			},
		},
		{
			"part_index": int64(2), "storage_mode": "EXTERNAL_URI", "mime_type": "application/pdf",
			"uri": "https://example.com/doc.pdf",
		},
		textPart("be brief", 0),
	}
	if diff := cmp.Diff(want, contentPartsOf(t, row)); diff != "" {
		t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
	}
	wantUploads := []uploadAttempt{
		{Path: textObject, ContentType: "text/plain; charset=utf-8", Data: []byte(longText)},
		{Path: imageObject, ContentType: "image/png", Data: png},
	}
	if diff := cmp.Diff(wantUploads, off.uploads(), ignoreDeadlines); diff != "" {
		t.Errorf("uploads mismatch (-want +got):\n%s", diff)
	}
	if row["is_truncated"] != false {
		t.Errorf("is_truncated = %v, want false: nothing was cut", row["is_truncated"])
	}
}

func TestGCSOffloading_WithoutAnOffloaderEverythingStaysInline(t *testing.T) {
	// A small MaxInlineTextSize alone does not turn offloading on.
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) { c.MaxInlineTextSize = 10 })
	longText := strings.Repeat("x", 100)
	req := &model.LLMRequest{Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{
		{Text: longText},
		{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}},
		{FileData: &genai.FileData{FileURI: "gs://bucket/a.pdf", MIMEType: "application/pdf"}},
	}}}}

	row := logRequest(t, p, requests, mCtx, req)

	want := []map[string]any{
		textPart(longText, 0),
		inlineImagePart(1),
		{"part_index": int64(2), "storage_mode": "EXTERNAL_URI", "mime_type": "application/pdf", "uri": "gs://bucket/a.pdf"},
	}
	if diff := cmp.Diff(want, contentPartsOf(t, row)); diff != "" {
		t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
	}
}

func TestGCSOffloading_ContentFormatterRunsBeforeTheUpload(t *testing.T) {
	off := &fakeOffloader{}
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
		c.MaxInlineTextSize = 20
		// This formatter hides a secret in the text of the request.
		c.ContentFormatter = func(payload any, _ string) (any, error) {
			req, ok := payload.(*model.LLMRequest)
			if !ok {
				return payload, nil
			}
			out := &model.LLMRequest{}
			for _, content := range req.Contents {
				var parts []*genai.Part
				for _, part := range content.Parts {
					parts = append(parts, &genai.Part{Text: strings.ReplaceAll(part.Text, "sk-secret-123", "[HIDDEN]")})
				}
				out.Contents = append(out.Contents, &genai.Content{Role: content.Role, Parts: parts})
			}
			return out, nil
		}
	})
	text := "my key is sk-secret-123 " + strings.Repeat("padding ", 20)

	row := logRequest(t, p, requests, mCtx, &model.LLMRequest{Contents: []*genai.Content{textContent("user", text)}})

	uploads := off.uploads()
	if len(uploads) != 1 {
		t.Fatalf("got %d uploads, want 1", len(uploads))
	}
	if got := string(uploads[0].Data); strings.Contains(got, "sk-secret-123") || !strings.Contains(got, "[HIDDEN]") {
		t.Errorf("the uploaded text is %q, want the formatter's version", got)
	}
	if got := contentPartsOf(t, row)[0]["text"].(string); strings.Contains(got, "sk-secret-123") || !strings.Contains(got, "[HIDDEN]") {
		t.Errorf("the preview is %q, want the formatter's version", got)
	}
}

func TestGCSOffloading_UploadFailureKeepsThePartsInline(t *testing.T) {
	off := &fakeOffloader{hook: func(context.Context, GCSUploadRequest) error {
		return errors.New("storage unavailable: SECRET-ERR-MARKER")
	}}
	logger := &recordingLogger{}
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
		c.MaxInlineTextSize = 20
		c.Logger = logger
	})
	longText := "SECRET-TEXT-MARKER" + strings.Repeat("-", 50)
	req := &model.LLMRequest{Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{
		{Text: longText},
		{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("SECRET-IMAGE-MARKER")}},
	}}}}

	row := logRequest(t, p, requests, mCtx, req)

	failedImage := inlineImagePart(1)
	failedImage["text"] = uploadFailedText
	want := []map[string]any{textPart(longText, 0), failedImage}
	if diff := cmp.Diff(want, contentPartsOf(t, row)); diff != "" {
		t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
	}
	if row["is_truncated"] != false {
		t.Errorf("is_truncated = %v, want false", row["is_truncated"])
	}
	if !logger.contains("GCS upload failed") || !logger.contains("error type *errors.errorString") {
		t.Errorf("the log lacks the failure:\n%s", logger)
	}
	for _, secret := range []string{"SECRET-TEXT-MARKER", "SECRET-IMAGE-MARKER", "SECRET-ERR-MARKER"} {
		if logger.contains(secret) {
			t.Errorf("the log contains %q:\n%s", secret, logger)
		}
	}
}

func TestGCSOffloading_FailedUploadOfCutTextKeepsTheCut(t *testing.T) {
	off := &fakeOffloader{hook: func(context.Context, GCSUploadRequest) error { return errors.New("storage unavailable") }}
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
		c.MaxContentLen = 50
		c.Logger = &recordingLogger{}
	})
	text := strings.Repeat("0123456789", 12)

	row := logRequest(t, p, requests, mCtx, &model.LLMRequest{Contents: []*genai.Content{textContent("user", text)}})

	want := []map[string]any{textPart(text[:50]+"...[TRUNCATED]", 0)}
	if diff := cmp.Diff(want, contentPartsOf(t, row)); diff != "" {
		t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
	}
	if row["is_truncated"] != true {
		t.Errorf("is_truncated = %v, want true: the row holds a cut of the text", row["is_truncated"])
	}
}

func TestGCSOffloading_TextCutByMaxContentLenIsOffloadedInFull(t *testing.T) {
	off := &fakeOffloader{}
	// The text is within the inline limit, but a row cannot keep all of it
	// inline: MaxContentLen would cut it.
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) { c.MaxContentLen = 50 })
	text := strings.Repeat("0123456789", 12)

	row := logRequest(t, p, requests, mCtx, &model.LLMRequest{Contents: []*genai.Content{textContent("user", text)}})

	uploads := off.uploads()
	if len(uploads) != 1 || string(uploads[0].Data) != text {
		t.Fatalf("uploads = %v, want one with all of the text", uploads)
	}
	part := contentPartsOf(t, row)[0]
	if part["storage_mode"] != "GCS_REFERENCE" || part["text"] != text+"... [OFFLOADED]" {
		t.Errorf("the part is %v, want a reference with the preview of the whole text", part)
	}
	if row["is_truncated"] != false {
		t.Errorf("is_truncated = %v, want false: the full text was offloaded", row["is_truncated"])
	}
}

func TestGCSOffloading_RowsWithoutContentUploadNothing(t *testing.T) {
	off := &fakeOffloader{}
	p, requests, mCtx := offloadingPlugin(t, off, nil)
	// An LLM_ERROR row has no content, although the request holds an image.
	req := &model.LLMRequest{Model: "gemini-test", Contents: []*genai.Content{imageContent(1, 2, 3)}}
	if _, err := p.OnModelErrorCallback()(mCtx, req, errors.New("model unavailable")); err != nil {
		t.Fatalf("OnModelErrorCallback() error: %v", err)
	}
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationError))

	row := onlyRowOfType(t, rows, EventLLMError)
	if row["status"] != "ERROR" || row["error_message"] != "model unavailable" {
		t.Errorf("status = %v, error_message = %v; want ERROR and the error", row["status"], row["error_message"])
	}
	if _, ok := row[columnAttributes]; !ok {
		t.Error("the row has no attributes")
	}
	if parts, ok := row[columnContentParts]; ok {
		t.Errorf("the row has content_parts %v, want none", parts)
	}
	if got := off.uploads(); len(got) != 0 {
		t.Errorf("got uploads %v, want none", got)
	}
}

func TestGCSOffloading_PanickingOffloaderLosesNoRows(t *testing.T) {
	off := &fakeOffloader{hook: func(context.Context, GCSUploadRequest) error { panic("PANIC-MARKER") }}
	logger := &recordingLogger{}
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) { c.Logger = logger })

	row := logRequest(t, p, requests, mCtx, &model.LLMRequest{Contents: []*genai.Content{imageContent(1)}})

	failedImage := inlineImagePart(0)
	failedImage["text"] = uploadFailedText
	if diff := cmp.Diff([]map[string]any{failedImage}, contentPartsOf(t, row)); diff != "" {
		t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
	}
	if !logger.contains("offloader panicked (string)") {
		t.Errorf("the log does not say that the offloader panicked:\n%s", logger)
	}
	if logger.contains("PANIC-MARKER") {
		t.Errorf("the log contains the panic value:\n%s", logger)
	}
}

func TestGCSOffloading_ObjectNamesTellTheMessagesApart(t *testing.T) {
	off := &fakeOffloader{}
	p, requests, mCtx := offloadingPlugin(t, off, nil)
	// Every message has its image at part index 0. A nil message keeps its
	// number, and the system instruction comes after all of the messages.
	req := &model.LLMRequest{
		Contents: []*genai.Content{imageContent(1), nil, imageContent(3)},
		Config:   &genai.GenerateContentConfig{SystemInstruction: imageContent(4)},
	}

	row := logRequest(t, p, requests, mCtx, req)

	base := fmt.Sprintf("%s/%v/%v_%v", rowDate(row), row["trace_id"], row["span_id"], row[eventIDField])
	want := map[string][]byte{
		base + "_c0_p0.png": {1},
		base + "_c2_p0.png": {3},
		base + "_c3_p0.png": {4},
	}
	got := map[string][]byte{}
	for _, u := range off.uploads() {
		got[u.Path] = u.Data
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("uploaded objects mismatch (-want +got):\n%s", diff)
	}
}

// TestGCSOffloading_CallerChangesItsMessagesWhileRowsAreWritten is the race
// test of the plugin. Once a callback has returned, the caller owns its
// messages again and may change them while the batch processor writes the row
// on another goroutine. Run with -race, the test fails if the row, or the
// payload queued with it, still shares memory with the caller.
func TestGCSOffloading_CallerChangesItsMessagesWhileRowsAreWritten(t *testing.T) {
	var passes atomic.Int64
	off := &fakeOffloader{hook: func(context.Context, GCSUploadRequest) error {
		// Hold the upload until the caller has been at work on its messages
		// for a while. The offloader reads the data after this.
		for deadline := time.Now().Add(2 * time.Second); passes.Load() < 20 && time.Now().Before(deadline); {
			time.Sleep(time.Millisecond)
		}
		return nil
	}}
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) { c.MaxInlineTextSize = 50 })

	pristineBlob := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	blob := bytes.Clone(pristineBlob)
	longText := strings.Repeat("abcdefghij", 30)
	longPart := &genai.Part{Text: longText}
	response := map[string]any{"k": 1000}
	req := &model.LLMRequest{Contents: []*genai.Content{
		{Role: "user", Parts: []*genai.Part{longPart, {InlineData: &genai.Blob{MIMEType: "image/png", Data: blob}}}},
		{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: response}}}},
	}}
	args := map[string]any{"q": "pristine"}
	var errs callbackErrors
	errs.add(p.BeforeModelCallback()(mCtx, req))
	errs.add(p.BeforeToolCallback()(mCtx, &fakeTool{name: "lookup"}, args))
	errs.check(t)

	// The callbacks have returned, so the caller is free to change its
	// messages. Its changes never stop until the rows are written.
	stop := make(chan struct{})
	var mutator sync.WaitGroup
	mutator.Go(func() {
		for {
			select {
			case <-stop:
				return
			default:
			}
			for i := range blob {
				blob[i] = 0xEE
			}
			longPart.Text = "MUTATED"
			response["k"] = -1
			args["q"] = "mutated"
			passes.Add(1)
			runtime.Gosched()
		}
	})
	p.AfterRunCallback()(mCtx)
	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	close(stop)
	mutator.Wait()

	// What was uploaded is what the messages held when the callback ran...
	got := map[string][]byte{}
	for _, u := range off.uploads() {
		got[u.Path[strings.LastIndex(u.Path, "_c"):]] = u.Data
	}
	want := map[string][]byte{"_c0_p0.txt": []byte(longText), "_c0_p1.png": pristineBlob}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("uploaded data mismatch (-want +got):\n%s", diff)
	}
	// ... and so is what the rows hold.
	parts := contentPartsOf(t, onlyRowOfType(t, rows, EventLLMRequest))
	if len(parts) != 3 {
		t.Fatalf("got %d content_parts, want 3: %v", len(parts), parts)
	}
	if want := longText[:200] + "... [OFFLOADED]"; parts[0]["text"] != want {
		t.Errorf("the text preview is %q, want %q", parts[0]["text"], want)
	}
	var attrs any
	if err := json.Unmarshal([]byte(parts[2]["part_attributes"].(string)), &attrs); err != nil {
		t.Fatalf("part_attributes is not JSON: %v", err)
	}
	checkJSON(t, "part_attributes", attrs, `{"function_response":{"name":"lookup","response":{"k":1000}}}`)
	toolContent := jsonObjectColumn(t, onlyRowOfType(t, rows, EventToolStarting), columnContent)
	checkJSONAt(t, toolContent, `{"q":"pristine"}`, "args")
}

func TestGCSOffloading_ManyCallersAndFlushes(t *testing.T) {
	const goroutines, perGoroutine = 6, 20
	off := &fakeOffloader{}
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
		c.BatchSize = 7
		c.MaxInlineTextSize = 64
	})
	var callers sync.WaitGroup
	for g := range goroutines {
		callers.Go(func() {
			for i := range perGoroutine {
				marker := fmt.Sprintf("g%02d-i%02d", g, i)
				req := &model.LLMRequest{Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{
					{Text: marker + strings.Repeat(".", 100)},
					{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{byte(g), byte(i)}}},
				}}}}
				if _, err := p.BeforeModelCallback()(mCtx, req); err != nil {
					t.Errorf("BeforeModelCallback() error: %v", err)
				}
			}
		})
	}
	callers.Wait()
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, func(rows []map[string]any) bool {
		return len(rowsOfType(rows, EventLLMRequest)) == goroutines*perGoroutine
	})

	uploaded := off.byPath()
	if got, want := len(off.uploads()), 2*goroutines*perGoroutine; got != want || len(uploaded) != want {
		t.Errorf("%d uploads to %d distinct objects, want %d of each", got, len(uploaded), want)
	}
	for _, row := range rowsOfType(rows, EventLLMRequest) {
		parts := contentPartsOf(t, row)
		if len(parts) != 2 {
			t.Errorf("a row has %d content_parts, want 2: %v", len(parts), parts)
			continue
		}
		textPath := strings.TrimPrefix(fmt.Sprint(parts[0]["uri"]), "gs://fake-bucket/")
		imagePath := strings.TrimPrefix(fmt.Sprint(parts[1]["uri"]), "gs://fake-bucket/")
		// The row names its own objects, and the objects hold its own data.
		wantPrefix := fmt.Sprintf("%s/%v/%v_%v_c0_p", rowDate(row), row["trace_id"], row["span_id"], row[eventIDField])
		if !strings.HasPrefix(textPath, wantPrefix) || !strings.HasPrefix(imagePath, wantPrefix) {
			t.Errorf("the objects of the row are %q and %q, want names that start with %q", textPath, imagePath, wantPrefix)
		}
		marker := strings.SplitN(parts[0]["text"].(string), ".", 2)[0]
		var g, i int
		if _, err := fmt.Sscanf(marker, "g%d-i%d", &g, &i); err != nil {
			t.Errorf("the preview %q of a row does not start with a marker", parts[0]["text"])
			continue
		}
		if data := uploaded[textPath].Data; !bytes.HasPrefix(data, []byte(marker)) {
			t.Errorf("the text object of %s holds %.20q", marker, data)
		}
		if data := uploaded[imagePath].Data; !bytes.Equal(data, []byte{byte(g), byte(i)}) {
			t.Errorf("the image object of %s holds %v, want %v", marker, data, []byte{byte(g), byte(i)})
		}
	}
}

// storageClientOptions returns the options that point the Cloud Storage client
// of a plugin at fs, and the transport that records what that client does.
func storageClientOptions(fs *fakeStorage) ([]option.ClientOption, *recordingTransport) {
	rt := &recordingTransport{inner: &http.Transport{}}
	return append(fs.clientOptions(), option.WithHTTPClient(&http.Client{Transport: rt})), rt
}

func TestGCSOffloading_BucketNameBuildsAnOffloader(t *testing.T) {
	fs := newFakeStorage(t)
	opts, _ := storageClientOptions(fs)
	// There is no GCSOffloader: the plugin builds one for the bucket.
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.GCSBucketName = "test-bucket"
		c.GCSPathPrefix = "agent-logs"
		c.ConnectionID = "us.conn"
		c.ClientOptions = opts
	})
	png := []byte{0x89, 'P', 'N', 'G'}

	row := logRequest(t, p, requests, mCtx, &model.LLMRequest{Contents: []*genai.Content{imageContent(png...)}})

	object := fmt.Sprintf("agent-logs/%s/%v/%v_%v_c0_p0.png", rowDate(row), row["trace_id"], row["span_id"], row[eventIDField])
	uri := "gs://test-bucket/" + object
	want := []map[string]any{{
		"part_index": int64(0), "storage_mode": "GCS_REFERENCE", "mime_type": "image/png",
		"text": "[MEDIA OFFLOADED]", "uri": uri,
		"object_ref": map[string]any{
			"uri": uri, "authorizer": "us.conn",
			"details": `{"gcs_metadata":{"content_type":"image/png","size":4}}`,
		},
	}}
	if diff := cmp.Diff(want, contentPartsOf(t, row)); diff != "" {
		t.Errorf("content_parts mismatch (-want +got):\n%s", diff)
	}
	got, ok := fs.object("test-bucket", object)
	if !ok {
		t.Fatalf("the object %q was not stored; got %v", object, fs.objects)
	}
	if got.contentType != "image/png" || !bytes.Equal(got.data, png) {
		t.Errorf("stored object = {%q, %x}, want {image/png, %x}", got.contentType, got.data, png)
	}
}

// TestGCSOffloading_ClosesTheOwnedOffloaderAfterTheProcessor checks the order
// in which the plugin closes what it owns: the batch processor uploads while
// it flushes, so closing the Cloud Storage client first would lose the
// uploads of the last flush.
func TestGCSOffloading_ClosesTheOwnedOffloaderAfterTheProcessor(t *testing.T) {
	fs := newFakeStorage(t)
	opts, rt := storageClientOptions(fs)
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.GCSBucketName = "test-bucket"
		c.ClientOptions = opts
		// Nothing but closing the plugin writes the row.
		c.DisableFlushOnRunEnd = true
		c.BatchFlushIntv = time.Hour
	})
	if _, err := p.BeforeModelCallback()(mCtx, &model.LLMRequest{Contents: []*genai.Content{imageContent(1)}}); err != nil {
		t.Fatalf("BeforeModelCallback() error: %v", err)
	}

	if err := p.Close(); err != nil {
		t.Fatalf("Close() error: %v", err)
	}

	rows := collectRows(t, requests, func(rows []map[string]any) bool { return len(rowsOfType(rows, EventLLMRequest)) == 1 })
	parts := contentPartsOf(t, onlyRowOfType(t, rows, EventLLMRequest))
	if len(parts) != 1 || parts[0]["storage_mode"] != "GCS_REFERENCE" {
		t.Errorf("content_parts = %v, want the image to be uploaded by the last flush", parts)
	}
	events := rt.recorded()
	upload, closed := slices.Index(events, "request POST"), slices.Index(events, "close")
	if upload < 0 || closed < 0 || upload > closed {
		t.Errorf("the Cloud Storage client saw %v, want its upload before it is closed", events)
	}
}

func TestGCSOffloading_SuppliedOffloaderWinsAndIsNotClosed(t *testing.T) {
	fs := newFakeStorage(t)
	opts, rt := storageClientOptions(fs)
	off := &fakeOffloader{}
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
		c.GCSBucketName = "ignored-bucket"
		c.ClientOptions = opts
	})

	row := logRequest(t, p, requests, mCtx, &model.LLMRequest{Contents: []*genai.Content{imageContent(1)}})
	if err := p.Close(); err != nil {
		t.Fatalf("Close() error: %v", err)
	}

	if uri := fmt.Sprint(contentPartsOf(t, row)[0]["uri"]); !strings.HasPrefix(uri, "gs://fake-bucket/") {
		t.Errorf("uri = %q, want an object of the supplied offloader", uri)
	}
	if got := off.closeCount(); got != 0 {
		t.Errorf("the supplied offloader was closed %d times; it belongs to the caller", got)
	}
	if got := rt.recorded(); len(got) != 0 {
		t.Errorf("the Cloud Storage client was used (%v) although an offloader was supplied", got)
	}
}

func TestGCSOffloading_OffloaderCreationFailureClosesTheWriteClient(t *testing.T) {
	t.Setenv("STORAGE_EMULATOR_HOST", "")
	ctx := context.Background()
	conn, err := grpc.NewClient("passthrough:///unused", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("grpc.NewClient() error: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatalf("NewBigQueryWriteClient() error: %v", err)
	}
	cfg := DefaultConfig()
	cfg.ProjectID = "test-project"
	cfg.GCSBucketName = "test-bucket"
	cfg.Logger = &recordingLogger{}
	// The credentials file is not there, so no Cloud Storage client can be made.
	cfg.ClientOptions = []option.ClientOption{option.WithCredentialsFile(filepath.Join(t.TempDir(), "missing.json"))}

	p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, nil, writeClient)

	if err == nil {
		_ = p.Close()
		t.Fatal("NewBigQueryAgentAnalyticsPluginWithClients() succeeded, want an error")
	}
	if !strings.Contains(err.Error(), "failed to create GCS offloader") {
		t.Errorf("error = %v, want one about the GCS offloader", err)
	}
	// The write client is the plugin's from the call on, so a failed
	// construction closes it, and closing it once more fails.
	if err := writeClient.Close(); err == nil {
		t.Error("the write client is still open after the failed construction")
	}
}

func TestExtensionForMIME(t *testing.T) {
	// What the MIME tables of the host say is up to the host, so these types
	// are made up. One has an extension that is unsafe in an object name,
	// which sorts first, and one that is safe; the other has only an unsafe
	// one.
	for ext, typ := range map[string]string{
		".aabad/x": "application/x-agentanalytics-mixed",
		".okext":   "application/x-agentanalytics-mixed",
		".bad/ext": "application/x-agentanalytics-unsafe",
	} {
		if err := mime.AddExtensionType(ext, typ); err != nil {
			t.Fatalf("mime.AddExtensionType(%q, %q) error: %v", ext, typ, err)
		}
	}
	tests := []struct {
		mimeType string
		want     string
	}{
		{"image/png", ".png"},
		{"image/jpeg", ".jpg"},
		{"image/gif", ".gif"},
		{"image/webp", ".webp"},
		{"application/pdf", ".pdf"},
		{"application/json", ".json"},
		{"audio/mpeg", ".mp3"},
		{"audio/wav", ".wav"},
		{"video/mp4", ".mp4"},
		{"video/webm", ".webm"},
		{"text/plain", ".txt"},
		// Case, blanks and parameters do not matter.
		{"IMAGE/PNG", ".png"},
		{" image/png ", ".png"},
		{"text/plain; charset=utf-8", ".txt"},
		{"image/jpeg;q=0.5", ".jpg"},
		// image/jpeg is where the table above differs from the answer of the
		// host, so these only pass if the table is asked with a clean type.
		{"IMAGE/JPEG", ".jpg"},
		{" image/jpeg ", ".jpg"},
		{"Image/Jpeg ; charset=x", ".jpg"},
		// An unsafe extension is skipped.
		{"application/x-agentanalytics-mixed", ".okext"},
		{"application/x-agentanalytics-unsafe", ".bin"},
		// Types without an extension.
		{"application/x-no-such-type", ".bin"},
		{"not a mime type", ".bin"},
		{"", ".bin"},
	}
	for _, tt := range tests {
		t.Run(tt.mimeType, func(t *testing.T) {
			if got := extensionForMIME(tt.mimeType); got != tt.want {
				t.Errorf("extensionForMIME(%q) = %q, want %q", tt.mimeType, got, tt.want)
			}
		})
	}
}

func TestIsSafeExtension(t *testing.T) {
	tests := []struct {
		ext  string
		want bool
	}{
		{".png", true},
		{".PNG", true},
		{".7z", true},
		{".a", true},
		{"." + strings.Repeat("a", 16), true},
		{"." + strings.Repeat("a", 17), false},
		{"", false},
		{".", false},
		{"png", false},
		{".tar.gz", false},
		{".p/ng", false},
		{".p ng", false},
		{".png\n", false},
		{".pñg", false},
		{".p_g", false},
		{".p-g", false},
		// The edges of the character classes, and the characters next to them.
		{".z", true},
		{".A", true},
		{".Z", true},
		{".0", true},
		{".9", true},
		{".`", false},
		{".{", false},
		{".@", false},
		{".[", false},
		{"./", false},
		{".:", false},
	}
	for _, tt := range tests {
		t.Run(tt.ext, func(t *testing.T) {
			if got := isSafeExtension(tt.ext); got != tt.want {
				t.Errorf("isSafeExtension(%q) = %v, want %v", tt.ext, got, tt.want)
			}
		})
	}
}

func TestPathSegment(t *testing.T) {
	tests := []struct {
		name, id, want string
	}{
		{"a trace id", "4bf92f3577b34da6a3ce929d0e0e4736", "4bf92f3577b34da6a3ce929d0e0e4736"},
		{"all the safe characters", "aZ09-_.x", "aZ09-_.x"},
		{"empty", "", "fallback"},
		{"a slash", "a/b", "a_b" + pathSegmentHashSuffix("a/b")},
		{"a path", "../x", ".._x" + pathSegmentHashSuffix("../x")},
		{"blanks and control characters", "a b\tc\nd", "a_b_c_d" + pathSegmentHashSuffix("a b\tc\nd")},
		{"one underscore per rune", "é?*", "___" + pathSegmentHashSuffix("é?*")},
		{"only unsafe characters", "///", "___" + pathSegmentHashSuffix("///")},
		{"the current directory", ".", "fallback" + pathSegmentHashSuffix(".")},
		{"the parent directory", "..", "fallback" + pathSegmentHashSuffix("..")},
		{"only dots", "....", "fallback" + pathSegmentHashSuffix("....")},
		{"dots next to a name", "a..", "a.."},
		{"a dot name that is not a path element", ".x", ".x"},
		{"the edges of the character classes", "azAZ09", "azAZ09"},
		{"just outside the character classes", "`{@[/: ", "_______" + pathSegmentHashSuffix("`{@[/: ")},
		{"exactly 128 safe ASCII bytes", strings.Repeat("a", 128), strings.Repeat("a", 128)},
		{"129 safe ASCII bytes is truncated with hash", strings.Repeat("a", 129), strings.Repeat("a", maxPathSegmentBytes-pathSegmentHashSuffixLen) + pathSegmentHashSuffix(strings.Repeat("a", 129))},
		{"tilde is outside the unchanged safe alphabet", "a~b", "a_b" + pathSegmentHashSuffix("a~b")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := pathSegment(tt.id, "fallback"); got != tt.want {
				t.Errorf("pathSegment(%q, \"fallback\") = %q, want %q", tt.id, got, tt.want)
			}
			if len(tt.id) > 0 && len(pathSegment(tt.id, "fallback")) > maxPathSegmentBytes {
				t.Errorf("len(pathSegment(%q, \"fallback\")) = %d, exceeds %d", tt.id, len(pathSegment(tt.id, "fallback")), maxPathSegmentBytes)
			}
		})
	}
}

func TestCleanGCSPrefix(t *testing.T) {
	// "aa" and 260 three-byte runes (782 bytes): the cut at 685 bytes falls
	// inside the 228th rune and backs up to 683 bytes.
	cjk := "aa" + strings.Repeat("日", 260)
	tests := []struct {
		name, prefix, want string
	}{
		{"empty", "", ""},
		{"only slashes", "///", ""},
		{"leading and trailing slashes are trimmed", "/logs/tenant/", "logs/tenant"},
		{"spaces, equals signs and Unicode are kept", "tenant=acme/logs v1/テナント", "tenant=acme/logs v1/テナント"},
		{"three dots are a name", "a/.../b", "a/.../b"},
		{"an empty inner segment is kept", "logs//tenant", "logs//tenant"},
		{"a dot segment is kept", "a/./b", "a/./b"},
		{"a dot-dot segment is kept", "logs/../tenant", "logs/../tenant"},
		{"only dot segments are kept after trimming slashes", "/./../", "./.."},
		{"a backslash is kept", `logs\prod`, `logs\prod`},
		{"a tilde is reserved and rewritten", "~team/logs~v1", "_team/logs_v1" + pathSegmentHashSuffix("~team/logs~v1")},
		{"an ACME path that does not lead is kept", "logs/.well-known/acme-challenge", "logs/.well-known/acme-challenge"},
		{"a .well-known without acme-challenge is kept", ".well-known/x", ".well-known/x"},
		{"a .well-known/acme-challenge-x lookalike segment is kept", ".well-known/acme-challenge-x", ".well-known/acme-challenge-x"},
		{"a .well-known//acme-challenge with empty second segment is kept", ".well-known//acme-challenge", ".well-known//acme-challenge"},
		{"513 bytes (kept under 750-byte cap)", strings.Repeat("a", 513), strings.Repeat("a", 513)},
		{"exactly 750 bytes", strings.Repeat("a", 750), strings.Repeat("a", 750)},
		{"751 bytes", strings.Repeat("a", 751), strings.Repeat("a", 685) + pathSegmentHashSuffix(strings.Repeat("a", 751))},
		{"900 bytes", strings.Repeat("a", 900), strings.Repeat("a", 685) + pathSegmentHashSuffix(strings.Repeat("a", 900))},
		{"the cut backs up to a rune boundary", cjk, "aa" + strings.Repeat("日", 227) + pathSegmentHashSuffix(cjk)},
		{"a tab", "logs\tx", "logs_x" + pathSegmentHashSuffix("logs\tx")},
		{"a newline", "logs\nx", "logs_x" + pathSegmentHashSuffix("logs\nx")},
		{"a carriage return", "logs\rx", "logs_x" + pathSegmentHashSuffix("logs\rx")},
		{"DEL", "logs\x7fx", "logs_x" + pathSegmentHashSuffix("logs\x7fx")},
		{"a C1 control character", "logs\u009bx", "logs_x" + pathSegmentHashSuffix("logs\u009bx")},
		{"next line", "logs\u0085x", "logs_x" + pathSegmentHashSuffix("logs\u0085x")},
		{"line separator", "logs\u2028x", "logs_x" + pathSegmentHashSuffix("logs\u2028x")},
		{"paragraph separator", "logs\u2029x", "logs_x" + pathSegmentHashSuffix("logs\u2029x")},
		{"a byte that is not UTF-8", "logs/\xff", "logs/_" + pathSegmentHashSuffix("logs/\xff")},
		{"the reserved ACME path", ".well-known/acme-challenge/x", "_well-known/acme-challenge/x" + pathSegmentHashSuffix(".well-known/acme-challenge/x")},
		{"the reserved ACME path in upper case", ".WELL-KNOWN/ACME-CHALLENGE", "_well-known/ACME-CHALLENGE" + pathSegmentHashSuffix(".WELL-KNOWN/ACME-CHALLENGE")},
		{"the reserved ACME path with leading and trailing slashes", "//.well-known/acme-challenge//", "_well-known/acme-challenge" + pathSegmentHashSuffix(".well-known/acme-challenge")},
		{"the reserved ACME path with Kelvin sign", ".well-\u212anown/acme-challenge/x", "_well-known/acme-challenge/x" + pathSegmentHashSuffix(".well-\u212anown/acme-challenge/x")},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cleanGCSPrefix(tt.prefix)
			if got != tt.want {
				t.Errorf("cleanGCSPrefix(%q) = %q, want %q", tt.prefix, got, tt.want)
			}
			if len(got) > maxGCSPrefixBytes || !utf8.ValidString(got) {
				t.Errorf("cleanGCSPrefix(%q) = %q (%d bytes), want valid UTF-8 of at most %d bytes", tt.prefix, got, len(got), maxGCSPrefixBytes)
			}
			trimmed := strings.Trim(tt.prefix, "/")
			if got == trimmed {
				if strings.Contains(got, "~") {
					t.Errorf("cleanGCSPrefix(%q) = %q kept '~' verbatim; '~' must be reserved for rewritten prefixes", tt.prefix, got)
				}
				if again := cleanGCSPrefix(got); again != got {
					t.Errorf("cleanGCSPrefix(%q) = %q was rewritten on second pass to %q", got, got, again)
				}
			} else {
				suffix := pathSegmentHashSuffix(trimmed)
				if !strings.HasSuffix(got, suffix) {
					t.Errorf("cleanGCSPrefix(%q) = %q, rewritten without the hash suffix of %q", tt.prefix, got, trimmed)
				}
				if base := strings.TrimSuffix(got, suffix); strings.Contains(base, "~") {
					t.Errorf("cleanGCSPrefix(%q) = %q contains '~' before its trailing hash suffix", tt.prefix, got)
				}
				if again := cleanGCSPrefix(got); again == got {
					t.Errorf("cleanGCSPrefix(%q) = %q collided with literal cleanGCSPrefix(%q) = %q; raw and rewritten prefix namespaces must be disjoint", tt.prefix, got, got, again)
				}
			}
		})
	}
	// Every byte that is not UTF-8 becomes '_', so only the hash tells such
	// prefixes apart.
	if a, b := cleanGCSPrefix("\xff"), cleanGCSPrefix("\xfe"); a == b {
		t.Errorf("cleanGCSPrefix(%q) and cleanGCSPrefix(%q) are both %q", "\xff", "\xfe", a)
	}
}

func TestTextPreview(t *testing.T) {
	tests := []struct {
		name, text, want string
	}{
		{"short", "hello", "hello" + textPreviewSuffix},
		{"empty", "", textPreviewSuffix},
		{"exactly the limit", strings.Repeat("a", 200), strings.Repeat("a", 200) + textPreviewSuffix},
		{"one over the limit", strings.Repeat("a", 201), strings.Repeat("a", 200) + textPreviewSuffix},
		{"long", strings.Repeat("a", 250), strings.Repeat("a", 200) + textPreviewSuffix},
		// The limit counts runes, so a rune is never cut in half.
		{"multibyte runes", strings.Repeat("é", 250), strings.Repeat("é", 200) + textPreviewSuffix},
		{"multibyte runes within the limit", strings.Repeat("é", 200), strings.Repeat("é", 200) + textPreviewSuffix},
		// BigQuery rejects invalid UTF-8, and a run of invalid bytes becomes
		// one replacement character.
		{"invalid UTF-8", "a\xff\xfeb", "a\uFFFDb" + textPreviewSuffix},
		{"invalid UTF-8 at the cut", strings.Repeat("a", 199) + "\xff\xfe tail", strings.Repeat("a", 199) + "\uFFFD" + textPreviewSuffix},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := textPreview(tt.text)
			if got != tt.want {
				t.Errorf("textPreview() = %q, want %q", got, tt.want)
			}
			if !utf8.ValidString(got) {
				t.Errorf("textPreview() = %q is not valid UTF-8", got)
			}
		})
	}
}

func TestObjectRef(t *testing.T) {
	tests := []struct {
		name       string
		authorizer string
		want       map[string]any
	}{
		{
			name:       "with an authorizer",
			authorizer: "us.conn",
			want:       map[string]any{"uri": "gs://b/o.png", "details": "{}", "authorizer": "us.conn"},
		},
		{
			name: "without an authorizer",
			want: map[string]any{"uri": "gs://b/o.png", "details": "{}"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// There is never a version, as adk-python writes none.
			if diff := cmp.Diff(tt.want, objectRef("gs://b/o.png", "{}", tt.authorizer)); diff != "" {
				t.Errorf("objectRef() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGCSMetadataDetails(t *testing.T) {
	tests := []struct {
		contentType string
		size        int
		want        string
	}{
		{"image/png", 7, `{"gcs_metadata":{"content_type":"image/png","size":7}}`},
		{"text/plain", 0, `{"gcs_metadata":{"content_type":"text/plain","size":0}}`},
		{"video/mp4", 123456789, `{"gcs_metadata":{"content_type":"video/mp4","size":123456789}}`},
	}
	for _, tt := range tests {
		if got := gcsMetadataDetails(tt.contentType, tt.size); got != tt.want {
			t.Errorf("gcsMetadataDetails(%q, %d) = %s, want %s", tt.contentType, tt.size, got, tt.want)
		}
	}

	// Characters that JSON escapes still make valid JSON that reads back.
	const odd = `a"b\c<d>&e`
	var decoded struct {
		Metadata struct {
			ContentType string `json:"content_type"`
			Size        int    `json:"size"`
		} `json:"gcs_metadata"`
	}
	if err := json.Unmarshal([]byte(gcsMetadataDetails(odd, 5)), &decoded); err != nil {
		t.Fatalf("gcsMetadataDetails(%q, 5) is not JSON: %v", odd, err)
	}
	if decoded.Metadata.ContentType != odd || decoded.Metadata.Size != 5 {
		t.Errorf("gcsMetadataDetails(%q, 5) reads back as %+v", odd, decoded.Metadata)
	}
}

// newAddContentPlugin returns a plugin that can build the content columns of a
// row, which is all that addContent needs.
func newAddContentPlugin(t *testing.T, mutate func(*Config)) *analyticsPlugin {
	t.Helper()
	cfg := DefaultConfig()
	if mutate != nil {
		mutate(&cfg)
	}
	cfg = applyDefaults(cfg)
	denied, err := validateConfig(cfg)
	if err != nil {
		t.Fatalf("validateConfig() error: %v", err)
	}
	return &analyticsPlugin{config: cfg, deniedColumns: denied}
}

// TestAddContent_QueuedPayloadsAreCopies checks, without a second goroutine,
// the property that the race test checks with one: once addContent has
// returned, nothing in the row shares memory with the caller's message.
func TestAddContent_QueuedPayloadsAreCopies(t *testing.T) {
	p := newAddContentPlugin(t, func(c *Config) {
		c.GCSOffloader = &fakeOffloader{}
		c.MaxInlineTextSize = 10
	})
	blob := []byte{1, 2, 3}
	longText := strings.Repeat("x", 50)
	longPart := &genai.Part{Text: longText}
	imagePart := &genai.Part{InlineData: &genai.Blob{MIMEType: "image/png", Data: blob}}
	row := map[string]any{}

	p.addContent(row, EventUserMessageReceived, &eventData{content: &genai.Content{Role: "user", Parts: []*genai.Part{longPart, imagePart}}})
	// The caller goes on using its message.
	blob[0] = 0xEE
	longPart.Text = "changed"
	imagePart.InlineData.MIMEType = "image/gif"

	want := []pendingUpload{
		{slot: 0, contentOrd: 0, partIdx: 0, mimeType: "text/plain", isText: true, text: longText},
		{slot: 1, contentOrd: 0, partIdx: 1, mimeType: "image/png", data: []byte{1, 2, 3}},
	}
	got, _ := row[rawPartsKey].([]pendingUpload)
	if diff := cmp.Diff(want, got, cmp.AllowUnexported(pendingUpload{})); diff != "" {
		t.Errorf("queued uploads mismatch (-want +got):\n%s", diff)
	}
	parts, _ := row[columnContentParts].([]map[string]any)
	if len(parts) != 2 || parts[0]["text"] != longText || parts[1]["mime_type"] != "image/png" {
		t.Errorf("content_parts = %v, want the entries of the message as it was", parts)
	}
}

func TestAddContent_QueuesOnlyWhenOffloading(t *testing.T) {
	longText := strings.Repeat("x", 50)
	long := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: longText}, {InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1}}}}}
	short := textContent("user", "short")
	offload := func(c *Config) {
		c.GCSOffloader = &fakeOffloader{}
		c.MaxInlineTextSize = 10
	}
	tests := []struct {
		name    string
		mutate  func(*Config)
		content *genai.Content
		// wantParts and wantQueued say whether the row gets content_parts and
		// queued uploads; wantContent whether it gets the content column.
		wantParts, wantQueued, wantContent bool
	}{
		{name: "with an offloader", mutate: offload, content: long, wantParts: true, wantQueued: true, wantContent: true},
		{name: "without an offloader", mutate: func(c *Config) { c.MaxInlineTextSize = 10 }, content: long, wantParts: true, wantContent: true},
		{name: "text within the limit", mutate: offload, content: short, wantParts: true, wantContent: true},
		{
			name: "content_parts denied",
			mutate: func(c *Config) {
				offload(c)
				c.PayloadColumnDenylist = []string{"content_parts"}
			},
			content: long, wantContent: true,
		},
		{
			// The upload is about content_parts, not about content.
			name: "content denied",
			mutate: func(c *Config) {
				offload(c)
				c.PayloadColumnDenylist = []string{"content"}
			},
			content: long, wantParts: true, wantQueued: true,
		},
		{
			name: "multi-modal content disabled",
			mutate: func(c *Config) {
				offload(c)
				c.DisableMultiModalContent = true
			},
			content: long, wantContent: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newAddContentPlugin(t, tt.mutate)
			row := map[string]any{}

			p.addContent(row, EventUserMessageReceived, &eventData{content: tt.content})

			if _, ok := row[columnContentParts]; ok != tt.wantParts {
				t.Errorf("row has content_parts = %v, want %v", ok, tt.wantParts)
			}
			if _, ok := row[rawPartsKey]; ok != tt.wantQueued {
				t.Errorf("row has queued uploads = %v, want %v", ok, tt.wantQueued)
			}
			if _, ok := row[columnContent]; ok != tt.wantContent {
				t.Errorf("row has content = %v, want %v", ok, tt.wantContent)
			}
		})
	}
}

// TestLLMRequestContent_ContentOrdinals checks the numbers that tell apart the
// objects of the messages of a request.
func TestLLMRequestContent_ContentOrdinals(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{imageContent(1), nil, imageContent(3)},
		Config:   &genai.GenerateContentConfig{SystemInstruction: imageContent(4)},
	}
	c := &partsCollector{opts: partsOptions{maxLen: 1000, redact: true, offload: true, inlineLimit: 50}}

	llmRequestContent(req, 1000, true, c)

	// The nil message keeps its number, and the system instruction comes
	// after all of the messages. Slots count the entries of content_parts.
	want := []pendingUpload{
		{slot: 0, contentOrd: 0, partIdx: 0, mimeType: "image/png", data: []byte{1}},
		{slot: 1, contentOrd: 2, partIdx: 0, mimeType: "image/png", data: []byte{3}},
		{slot: 2, contentOrd: 3, partIdx: 0, mimeType: "image/png", data: []byte{4}},
	}
	if diff := cmp.Diff(want, c.pending, cmp.AllowUnexported(pendingUpload{})); diff != "" {
		t.Errorf("queued uploads mismatch (-want +got):\n%s", diff)
	}
}

// TestLLMRequestContent_Redaction checks that the summaries are redacted
// whenever the caller asks for it, with or without a parts collector.
func TestLLMRequestContent_Redaction(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "call it with api_key=SECRET123"}}}},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{Parts: []*genai.Part{{Text: "db password: hunter2"}}},
		},
	}
	tests := []struct {
		name       string
		redact     bool
		parts      *partsCollector
		wantSecret bool
	}{
		{name: "redacted without parts", redact: true},
		{name: "redacted with parts", redact: true, parts: &partsCollector{opts: partsOptions{maxLen: 1000, redact: true}}},
		{name: "not redacted without parts", wantSecret: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			content, _ := llmRequestContent(req, 1000, tt.redact, tt.parts)
			got, err := json.Marshal(content)
			if err != nil {
				t.Fatalf("json.Marshal(%v) error: %v", content, err)
			}
			for _, secret := range []string{"SECRET123", "hunter2"} {
				if has := strings.Contains(string(got), secret); has != tt.wantSecret {
					t.Errorf("content %s contains %q = %v, want %v", got, secret, has, tt.wantSecret)
				}
			}
			if !tt.wantSecret && !strings.Contains(string(got), redactedValue) {
				t.Errorf("content %s has no %q", got, redactedValue)
			}
		})
	}
}

// TestGCSOffloading_CloseReportsTheCloseErrors checks that Close returns what
// went wrong while it closed the clients of the plugin. Nothing fails the first
// time; closing the plugin again finds its BigQuery write client closed, which
// fails.
func TestGCSOffloading_CloseReportsTheCloseErrors(t *testing.T) {
	fs := newFakeStorage(t)
	opts, _ := storageClientOptions(fs)
	p, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
		c.GCSBucketName = "test-bucket"
		c.ClientOptions = opts
	})

	if err := p.Close(); err != nil {
		t.Fatalf("Close() error: %v", err)
	}

	err := p.Close()
	if err == nil {
		t.Fatal("closing the plugin again succeeded, want the error of closing the write client again")
	}
	if want := "failed to close BigQuery write client"; !strings.Contains(err.Error(), want) {
		t.Errorf("closing the plugin again returned %q, want an error that contains %q", err, want)
	}
}

// TestAddContent_PartAttributesFollowTheRedactionSetting checks that the
// content_parts of a row are redacted exactly when the plugin is configured to
// redact: the collector of the parts is given the setting of the plugin.
func TestAddContent_PartAttributesFollowTheRedactionSetting(t *testing.T) {
	const secret = "sk-test-secret-value"
	tests := []struct {
		name       string
		mutate     func(*Config)
		wantAPIKey string
	}{
		{name: "by default", wantAPIKey: redactedValue},
		{name: "redaction disabled", mutate: func(c *Config) { c.DisableSensitiveKeyRedaction = true }, wantAPIKey: secret},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newAddContentPlugin(t, tt.mutate)
			row := map[string]any{}
			content := &genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				Name:     "lookup",
				Response: map[string]any{"api_key": secret, "result": "ok"},
			}}}}

			p.addContent(row, EventUserMessageReceived, &eventData{content: content})

			parts := contentPartsOf(t, row)
			if len(parts) != 1 {
				t.Fatalf("content_parts = %v, want one entry", parts)
			}
			var attrs struct {
				FunctionResponse struct {
					Response map[string]any `json:"response"`
				} `json:"function_response"`
			}
			if err := json.Unmarshal([]byte(fmt.Sprint(parts[0]["part_attributes"])), &attrs); err != nil {
				t.Fatalf("part_attributes %v is not JSON: %v", parts[0]["part_attributes"], err)
			}
			want := map[string]any{"api_key": tt.wantAPIKey, "result": "ok"}
			if diff := cmp.Diff(want, attrs.FunctionResponse.Response); diff != "" {
				t.Errorf("the response in part_attributes mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestAddContent_TruncatedPartsMarkTheRow checks that a row is marked
// truncated when nothing but its content_parts lost content. The content
// column of these messages is a short summary that is never cut.
func TestAddContent_TruncatedPartsMarkTheRow(t *testing.T) {
	tests := []struct {
		name string
		part *genai.Part
		want bool
	}{
		{
			name: "a function response cut to the limit",
			part: &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: map[string]any{"data": strings.Repeat("x", 1000)}}},
			want: true,
		},
		{
			name: "a file whose URI carried credentials",
			part: &genai.Part{FileData: &genai.FileData{MIMEType: "image/png", FileURI: "https://user:pass@example.com/a.png"}},
			want: true,
		},
		{
			name: "a function response within the limit",
			part: &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "lookup", Response: map[string]any{"data": "x"}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newAddContentPlugin(t, func(c *Config) { c.MaxContentLen = 100 })
			row := map[string]any{}

			got := p.addContent(row, EventUserMessageReceived, &eventData{content: &genai.Content{Role: "user", Parts: []*genai.Part{tt.part}}})

			if got != tt.want {
				t.Errorf("addContent() = %v, want %v", got, tt.want)
			}
			// The content column is not what truncated the row.
			if content := fmt.Sprint(row[columnContent]); strings.Contains(content, truncationSuffix) {
				t.Errorf("the content column was truncated: %s", content)
			}
		})
	}
}

// TestResolveUploads_DefaultConcurrencyIsEight pins the default number of
// parallel uploads. TestResolveUploads_ConcurrencyIsBounded takes what it
// expects for that case from the constant, so it cannot notice the constant
// changing.
func TestResolveUploads_DefaultConcurrencyIsEight(t *testing.T) {
	const want = 8
	contents := make([]*genai.Content, 12)
	for i := range contents {
		contents[i] = imageContent(byte(i))
	}
	row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}, contents...)
	// The first uploads wait for each other, which shows that want of them
	// run at once; the pause that follows gives more a chance to start if the
	// default were higher. Once that wait fails the other uploads fail at
	// once, instead of waiting for their turn to fail.
	gate := newBarrier(want)
	var gateFailed atomic.Bool
	off := &fakeOffloader{hook: func(context.Context, GCSUploadRequest) error {
		if gateFailed.Load() {
			return errors.New("the first uploads did not run together")
		}
		if err := gate.wait(); err != nil {
			gateFailed.Store(true)
			return err
		}
		time.Sleep(10 * time.Millisecond)
		return nil
	}}
	bp := &BatchProcessor{config: Config{GCSOffloader: off, Logger: &recordingLogger{}}}

	bp.resolveUploads(context.Background(), false, []map[string]any{row})

	if got := off.maxConcurrent(); got != want {
		t.Errorf("%d uploads ran at the same time, want %d", got, want)
	}
	if got := len(off.uploads()); got != len(contents) {
		t.Errorf("%d uploads were made, want %d", got, len(contents))
	}
}

// TestAddContent_DefaultInlineLimitIs32KiB checks the threshold that applies
// when MaxInlineTextSize is not set: text of 32 KiB stays in the row, and one
// byte more is queued for upload.
func TestAddContent_DefaultInlineLimitIs32KiB(t *testing.T) {
	tests := []struct {
		name       string
		size       int
		wantQueued bool
	}{
		{name: "at the limit", size: 32 * 1024},
		{name: "one byte over the limit", size: 32*1024 + 1, wantQueued: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newAddContentPlugin(t, func(c *Config) {
				c.GCSOffloader = &fakeOffloader{}
				c.MaxInlineTextSize = 0
			})
			row := map[string]any{}

			p.addContent(row, EventUserMessageReceived, &eventData{content: textContent("user", strings.Repeat("x", tt.size))})

			if _, queued := row[rawPartsKey]; queued != tt.wantQueued {
				t.Errorf("a text of %d bytes is queued for upload = %v, want %v", tt.size, queued, tt.wantQueued)
			}
		})
	}
}

// TestApplyDefaults_GCSLimits pins the documented defaults of the limits
// of the offloading, and which values select them: any value that is not
// positive does.
func TestApplyDefaults_GCSLimits(t *testing.T) {
	const (
		wantInline  = 32 * 1024
		wantUploads = 8
		wantPending = int64(256 << 20)
	)
	tests := []struct {
		name            string
		inline, uploads int
		pending         int64
		wantInline      int
		wantUploads     int
		wantPending     int64
	}{
		{name: "zero values", wantInline: wantInline, wantUploads: wantUploads, wantPending: wantPending},
		{name: "negative values", inline: -1, uploads: -5, pending: -10, wantInline: wantInline, wantUploads: wantUploads, wantPending: wantPending},
		{name: "the smallest positive values", inline: 1, uploads: 1, pending: 1, wantInline: 1, wantUploads: 1, wantPending: 1},
		{name: "larger values", inline: 1 << 20, uploads: 64, pending: 512 << 20, wantInline: 1 << 20, wantUploads: 64, wantPending: 512 << 20},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := applyDefaults(Config{
				MaxInlineTextSize:        tt.inline,
				GCSMaxConcurrentUploads:  tt.uploads,
				GCSMaxPendingUploadBytes: tt.pending,
			})

			if got.MaxInlineTextSize != tt.wantInline {
				t.Errorf("MaxInlineTextSize = %d, want %d", got.MaxInlineTextSize, tt.wantInline)
			}
			if got.GCSMaxConcurrentUploads != tt.wantUploads {
				t.Errorf("GCSMaxConcurrentUploads = %d, want %d", got.GCSMaxConcurrentUploads, tt.wantUploads)
			}
			if got.GCSMaxPendingUploadBytes != tt.wantPending {
				t.Errorf("GCSMaxPendingUploadBytes = %d, want %d", got.GCSMaxPendingUploadBytes, tt.wantPending)
			}
		})
	}

	t.Run("DefaultConfig", func(t *testing.T) {
		got := DefaultConfig()

		if got.MaxInlineTextSize != wantInline {
			t.Errorf("MaxInlineTextSize = %d, want %d", got.MaxInlineTextSize, wantInline)
		}
		if got.GCSMaxConcurrentUploads != wantUploads {
			t.Errorf("GCSMaxConcurrentUploads = %d, want %d", got.GCSMaxConcurrentUploads, wantUploads)
		}
		if got.GCSUploadTimeout != gcsUploadTimeout {
			t.Errorf("GCSUploadTimeout = %v, want %v", got.GCSUploadTimeout, gcsUploadTimeout)
		}
		if got.GCSMaxPendingUploadBytes != wantPending {
			t.Errorf("GCSMaxPendingUploadBytes = %d, want %d", got.GCSMaxPendingUploadBytes, wantPending)
		}
	})
}

func TestGCSOffloading_RedactsSensitiveJSONBeforeUpload(t *testing.T) {
	off := &fakeOffloader{}
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
		c.MaxInlineTextSize = 20
	})
	secretPayload := `{"api_key":"sk-live-secret-999","note":"` + strings.Repeat("z", 80) + `"}`

	row := logRequest(t, p, requests, mCtx, &model.LLMRequest{Contents: []*genai.Content{textContent("user", secretPayload)}})

	uploads := off.uploads()
	if len(uploads) != 1 {
		t.Fatalf("got %d uploads, want 1", len(uploads))
	}
	if got := string(uploads[0].Data); strings.Contains(got, "sk-live-secret-999") || !strings.Contains(got, redactedValue) {
		t.Errorf("uploaded text = %q, want sensitive key redacted before GCS upload", got)
	}
	preview := contentPartsOf(t, row)[0]["text"].(string)
	if strings.Contains(preview, "sk-live-secret-999") || !strings.Contains(preview, redactedValue) {
		t.Errorf("inline preview = %q, want sensitive key redacted", preview)
	}
}

func TestGCSOffloading_OmitsOffloadedTextFromContentSummary(t *testing.T) {
	t.Run("successful upload omits offloaded text from USER_MESSAGE_RECEIVED and LLM_REQUEST summaries", func(t *testing.T) {
		off := &fakeOffloader{}
		p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
			c.MaxInlineTextSize = 40
			c.MaxContentLen = 60
		})
		offloadedBody := "OFFLOADED-BODY-" + strings.Repeat("A", 120)
		userMsg := &genai.Content{Role: "user", Parts: []*genai.Part{
			{Text: "keep-me"},
			{Text: offloadedBody},
		}}
		if _, err := p.OnUserMessageCallback()(mCtx, userMsg); err != nil {
			t.Fatalf("OnUserMessageCallback() error: %v", err)
		}
		sysOffloaded := "OFFLOADED-SYS-" + strings.Repeat("B", 120)
		req := &model.LLMRequest{
			Contents: []*genai.Content{userMsg},
			Config:   &genai.GenerateContentConfig{SystemInstruction: textContent("", sysOffloaded)},
		}
		if _, err := p.BeforeModelCallback()(mCtx, req); err != nil {
			t.Fatalf("BeforeModelCallback() error: %v", err)
		}
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))

		userRow := onlyRowOfType(t, rows, EventUserMessageReceived)
		userContent := jsonObjectColumn(t, userRow, columnContent)
		if got := userContent["text_summary"]; got != "keep-me" {
			t.Errorf("USER_MESSAGE_RECEIVED text_summary = %q, want %q (offloaded text omitted)", got, "keep-me")
		}
		if userRow["is_truncated"] != false {
			t.Errorf("USER_MESSAGE_RECEIVED is_truncated = %v, want false", userRow["is_truncated"])
		}

		llmRow := onlyRowOfType(t, rows, EventLLMRequest)
		llmContent := jsonObjectColumn(t, llmRow, columnContent)
		checkJSONAt(t, llmContent, `[{"role":"user","content":"keep-me"}]`, "prompt")
		if got := llmContent["system_prompt"]; got != "" {
			t.Errorf("LLM_REQUEST system_prompt = %q, want empty when system prompt was offloaded", got)
		}
		if llmRow["is_truncated"] != false {
			t.Errorf("LLM_REQUEST is_truncated = %v, want false", llmRow["is_truncated"])
		}
	})

	t.Run("successful upload omits offloaded text from LLM_RESPONSE while keeping non-text parts and usage", func(t *testing.T) {
		off := &fakeOffloader{}
		p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
			c.MaxInlineTextSize = 40
			c.MaxContentLen = 60
		})
		offloadedResp := "OFFLOADED-RESP-" + strings.Repeat("C", 120)
		resp := &model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{Text: offloadedResp},
					{FunctionCall: &genai.FunctionCall{Name: "lookup"}},
				},
			},
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     10,
				CandidatesTokenCount: 20,
				TotalTokenCount:      30,
			},
		}
		if _, err := p.AfterModelCallback()(mCtx, resp, nil); err != nil {
			t.Fatalf("AfterModelCallback() error: %v", err)
		}
		p.AfterRunCallback()(mCtx)

		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
		respRow := onlyRowOfType(t, rows, EventLLMResponse)
		respContent := jsonObjectColumn(t, respRow, columnContent)
		if got := respContent["response"]; got != "call: lookup" {
			t.Errorf("LLM_RESPONSE content.response = %q, want %q (offloaded text omitted)", got, "call: lookup")
		}
		checkJSONAt(t, respContent, `{"prompt":10,"completion":20,"total":30}`, "usage")
		if respRow["is_truncated"] != false {
			t.Errorf("LLM_RESPONSE is_truncated = %v, want false", respRow["is_truncated"])
		}
	})

	t.Run("failed upload retains truncated inline text in summary and marks row truncated", func(t *testing.T) {
		off := &fakeOffloader{hook: func(context.Context, GCSUploadRequest) error { return errors.New("boom") }}
		p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
			c.MaxInlineTextSize = 40
			c.MaxContentLen = 60
			c.Logger = &recordingLogger{}
		})
		offloadedBody := "OFFLOADED-BODY-" + strings.Repeat("A", 120)
		userMsg := &genai.Content{Role: "user", Parts: []*genai.Part{
			{Text: "keep-me"},
			{Text: offloadedBody},
		}}
		if _, err := p.BeforeModelCallback()(mCtx, &model.LLMRequest{Contents: []*genai.Content{userMsg}}); err != nil {
			t.Fatalf("BeforeModelCallback() error: %v", err)
		}
		offloadedResp := "OFFLOADED-RESP-" + strings.Repeat("R", 120)
		if _, err := p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: textContent("model", offloadedResp)}, nil); err != nil {
			t.Fatalf("AfterModelCallback() error: %v", err)
		}
		p.AfterRunCallback()(mCtx)
		rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))

		row := onlyRowOfType(t, rows, EventLLMRequest)
		llmContent := jsonObjectColumn(t, row, columnContent)
		promptJSON, _ := json.Marshal(llmContent["prompt"])
		if !strings.Contains(string(promptJSON), "OFFLOADED-BODY-") {
			t.Errorf("failed upload should retain inline text in prompt summary, got %s", promptJSON)
		}
		if row["is_truncated"] != true {
			t.Errorf("LLM_REQUEST is_truncated = %v, want true when failed upload fell back to truncated inline text", row["is_truncated"])
		}

		respRow := onlyRowOfType(t, rows, EventLLMResponse)
		respContent := jsonObjectColumn(t, respRow, columnContent)
		if got, _ := respContent["response"].(string); !strings.HasPrefix(got, "text: 'OFFLOADED-RESP-") || !strings.HasSuffix(got, "...[TRUNCATED]'") {
			t.Errorf("failed LLM_RESPONSE upload should restore inline text in content.response, got %q", got)
		}
		if respRow["is_truncated"] != true {
			t.Errorf("LLM_RESPONSE is_truncated = %v, want true when failed upload fell back to truncated inline text", respRow["is_truncated"])
		}
	})

	t.Run("batch of large offloaded texts keeps AppendRows request small", func(t *testing.T) {
		off := &fakeOffloader{}
		p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
			c.MaxInlineTextSize = 1024
		})
		msgText := strings.Repeat("X", 40*1024) // 40 KiB
		contents := make([]*genai.Content, 50)  // 2 MiB total text across one LLMRequest
		for i := range contents {
			contents[i] = textContent("user", msgText)
		}
		if _, err := p.BeforeModelCallback()(mCtx, &model.LLMRequest{Contents: contents}); err != nil {
			t.Fatalf("BeforeModelCallback() error: %v", err)
		}
		p.AfterRunCallback()(mCtx)

		var req *storagepb.AppendRowsRequest
		select {
		case req = <-requests:
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for the batch to be written")
		}
		size := len(req.GetArrowRows().GetRows().GetSerializedRecordBatch())
		// With offloaded text omitted from content, the batch is well under 200 KiB
		// instead of duplicating 2 MiB in the content column.
		if size > 256*1024 {
			t.Errorf("serialized record batch size = %d bytes, want <= 256 KiB when text is offloaded", size)
		}
	})

	t.Run("batch of 100 LLM_RESPONSE rows with 300 KiB offloaded text stays well below 20 MB", func(t *testing.T) {
		off := &fakeOffloader{}
		p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
			c.BatchSize = 100
			c.BatchFlushIntv = time.Hour
			c.MaxContentLen = 500 * 1024
			c.MaxInlineTextSize = 32 * 1024
		})
		respText := strings.Repeat("R", 300*1024) // 300 KiB per response
		for range 100 {
			if _, err := p.AfterModelCallback()(mCtx, &model.LLMResponse{Content: textContent("model", respText)}, nil); err != nil {
				t.Fatalf("AfterModelCallback() error: %v", err)
			}
		}

		var req *storagepb.AppendRowsRequest
		select {
		case req = <-requests:
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for the batch to be written")
		}
		size := len(req.GetArrowRows().GetRows().GetSerializedRecordBatch())
		// 100 x 300 KiB = 30 MiB if duplicated in content.response; omitting offloaded text keeps the batch < 512 KiB.
		if size > 512*1024 {
			t.Errorf("serialized LLM_RESPONSE record batch size = %d bytes, want <= 512 KiB when 300 KiB texts are offloaded", size)
		}
	})
}

func TestGCSOffloading_SkipsClientWhenContentPartsDisabled(t *testing.T) {
	t.Setenv("STORAGE_EMULATOR_HOST", "")
	badCreds := option.WithCredentialsFile(filepath.Join(t.TempDir(), "does-not-exist.json"))

	tests := []struct {
		name   string
		mutate func(*Config)
	}{
		{
			name:   "DisableMultiModalContent",
			mutate: func(c *Config) { c.DisableMultiModalContent = true },
		},
		{
			name:   "content_parts in PayloadColumnDenylist",
			mutate: func(c *Config) { c.PayloadColumnDenylist = []string{"content_parts"} },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := &recordingLogger{}
			p, _, _ := setupTestPluginWithConfig(t, func(c *Config) {
				c.GCSBucketName = "unused-bucket"
				// If newGCSOffloader were called, this invalid credentials file would fail construction.
				c.GCSClientOptions = []option.ClientOption{badCreds}
				c.Logger = logger
				tt.mutate(c)
			})
			if err := p.Close(); err != nil {
				t.Fatalf("Close() error: %v", err)
			}
			if !logger.contains("GCSBucketName \"unused-bucket\" is ignored") {
				t.Errorf("expected warning that GCSBucketName is ignored, got:\n%s", logger)
			}
		})
	}
}

func TestGCSOffloading_GCSClientOptionsOverridesClientOptions(t *testing.T) {
	fs := newFakeStorage(t)
	gcsOpts, rt := storageClientOptions(fs)
	badCreds := option.WithCredentialsFile(filepath.Join(t.TempDir(), "does-not-exist.json"))

	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.GCSBucketName = "test-bucket"
		// ClientOptions has invalid credentials that would fail Cloud Storage if used;
		// GCSClientOptions points cleanly at the fake storage server.
		c.ClientOptions = []option.ClientOption{badCreds}
		c.GCSClientOptions = gcsOpts
	})
	row := logRequest(t, p, requests, mCtx, &model.LLMRequest{Contents: []*genai.Content{imageContent(1, 2)}})
	if err := p.Close(); err != nil {
		t.Fatalf("Close() error: %v", err)
	}
	if got := contentPartsOf(t, row)[0]["storage_mode"]; got != "GCS_REFERENCE" {
		t.Errorf("storage_mode = %v, want GCS_REFERENCE", got)
	}
	if !slices.Contains(rt.recorded(), "request POST") {
		t.Errorf("GCSClientOptions transport was not used: %v", rt.recorded())
	}
}

func TestResolveUploads_BatchLevelUploadTimeout(t *testing.T) {
	contents := make([]*genai.Content, 6)
	for i := range contents {
		contents[i] = imageContent(byte(i))
	}
	row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}, contents...)
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		<-ctx.Done()
		return ctx.Err()
	}}
	bp := &BatchProcessor{
		config: Config{
			GCSOffloader:            off,
			GCSMaxConcurrentUploads: 2,
			GCSUploadTimeout:        60 * time.Millisecond,
			Logger:                  &recordingLogger{},
		},
	}

	start := time.Now()
	bp.resolveUploads(context.Background(), false, []map[string]any{row})
	elapsed := time.Since(start)

	// With a per-batch timeout of 60ms across 6 uploads at concurrency 2,
	// the entire batch finishes around 60ms rather than 3 * 60ms = 180ms.
	if elapsed >= 150*time.Millisecond {
		t.Errorf("resolveUploads took %v, want < 150ms (single batch-level timeout)", elapsed)
	}
	for i, part := range row[columnContentParts].([]map[string]any) {
		if part["text"] != uploadFailedText {
			t.Errorf("part %d text = %v, want %q", i, part["text"], uploadFailedText)
		}
	}
}

func TestGCSOffloading_PendingUploadBytesCap(t *testing.T) {
	logger := &recordingLogger{}
	bp := &BatchProcessor{
		config:                Config{GCSOffloader: &fakeOffloader{}, BatchSize: 100, Logger: logger},
		queue:                 make(chan map[string]any, 2),
		maxPendingUploadBytes: 100,
	}
	p := newAddContentPlugin(t, func(c *Config) {
		c.GCSOffloader = bp.config.GCSOffloader
		c.MaxInlineTextSize = 20
	})
	p.processor = bp

	// First 60-byte image fits under the 100-byte cap.
	row1 := map[string]any{}
	p.addContent(row1, EventLLMRequest, &eventData{content: &model.LLMRequest{Contents: []*genai.Content{imageContent(make([]byte, 60)...)}}})
	if _, ok := row1[rawPartsKey]; !ok {
		t.Fatal("row1 should be queued for upload (60 <= 100)")
	}
	if got := bp.pendingUploadBytes.Load(); got != 60 {
		t.Fatalf("pendingUploadBytes after row1 = %d, want 60", got)
	}
	bp.Append(row1)

	// Second 60-byte image exceeds the 100-byte cap (60 + 60 > 100): not queued, marked [UPLOAD FAILED].
	row2 := map[string]any{}
	p.addContent(row2, EventLLMRequest, &eventData{content: &model.LLMRequest{Contents: []*genai.Content{imageContent(make([]byte, 60)...)}}})
	if _, ok := row2[rawPartsKey]; ok {
		t.Fatal("row2 should not be queued when pending upload bytes cap is exceeded")
	}
	if got := contentPartsOf(t, row2)[0]["text"]; got != uploadFailedText {
		t.Errorf("row2 part text = %v, want %q when budget is exceeded", got, uploadFailedText)
	}
	if !logger.contains("pending GCS upload byte budget (100 bytes) exceeded") {
		t.Errorf("expected budget exhaustion warning in log, got:\n%s", logger)
	}
	if got := bp.pendingUploadBytes.Load(); got != 60 {
		t.Fatalf("pendingUploadBytes after rejected row2 = %d, want 60", got)
	}

	// Resolving row1 releases its 60 reserved bytes.
	drained := bp.drainQueue()
	bp.resolveUploads(context.Background(), false, drained)
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Fatalf("pendingUploadBytes after resolveUploads = %d, want 0", got)
	}

	// If a row with reserved bytes is dropped because the queue is full, its reservation is released.
	bp.Append(map[string]any{})
	bp.Append(map[string]any{})
	if len(bp.queue) != cap(bp.queue) {
		t.Fatalf("len(bp.queue) = %d, want full capacity %d", len(bp.queue), cap(bp.queue))
	}
	overflowRow := map[string]any{}
	p.addContent(overflowRow, EventLLMRequest, &eventData{content: &model.LLMRequest{Contents: []*genai.Content{imageContent(make([]byte, 40)...)}}})
	if got := bp.pendingUploadBytes.Load(); got != 40 {
		t.Fatalf("pendingUploadBytes before full-queue Append = %d, want 40", got)
	}
	bp.Append(overflowRow)
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes after full-queue dropped Append = %d, want 0", got)
	}
	_ = bp.drainQueue()

	// If a row with reserved bytes is dropped because the processor is closed, its reservation is released.
	bp.closed.Store(true)
	row3 := map[string]any{}
	p.addContent(row3, EventLLMRequest, &eventData{content: &model.LLMRequest{Contents: []*genai.Content{imageContent(make([]byte, 40)...)}}})
	if got := bp.pendingUploadBytes.Load(); got != 40 {
		t.Fatalf("pendingUploadBytes before dropped Append = %d, want 40", got)
	}
	bp.Append(row3)
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes after dropped Append = %d, want 0", got)
	}
}

func TestGCSOffloading_AfterRunFlushCapsUploadDeadline(t *testing.T) {
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		<-ctx.Done()
		return ctx.Err()
	}}
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
		c.BatchSize = 100
		c.BatchFlushIntv = time.Hour
		c.ShutdownTimeout = 200 * time.Millisecond
		c.GCSUploadTimeout = 900 * time.Millisecond
		c.Logger = &recordingLogger{}
	})
	if _, err := p.BeforeModelCallback()(mCtx, &model.LLMRequest{Contents: []*genai.Content{imageContent(1, 2, 3)}}); err != nil {
		t.Fatalf("BeforeModelCallback() error: %v", err)
	}

	start := time.Now()
	p.AfterRunCallback()(mCtx)
	elapsed := time.Since(start)
	// Uploads during AfterRunCallback must be capped to half of ShutdownTimeout (~100ms),
	// leaving the remaining budget for the fallback BigQuery append rather than waiting GCSUploadTimeout (900ms).
	if elapsed >= 350*time.Millisecond {
		t.Errorf("AfterRunCallback() took %v, want < 350ms (upload deadline should be capped by flush deadline)", elapsed)
	}

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	llmRow := onlyRowOfType(t, rows, EventLLMRequest)
	if got := contentPartsOf(t, llmRow)[0]["text"]; got != uploadFailedText {
		t.Errorf("fallback part text = %v, want %q", got, uploadFailedText)
	}
}

// TestGCSOffloading_RunEndsStopWaitingOnceStorageTimesOut checks that a Cloud
// Storage outage costs the runs of a long-lived plugin a single wait: once the
// uploads of one run end have timed out, the run ends after it do not wait for
// Cloud Storage but write their rows at once, with the parts that could not be
// uploaded in the form that a failed upload leaves.
func TestGCSOffloading_RunEndsStopWaitingOnceStorageTimesOut(t *testing.T) {
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		<-ctx.Done()
		return ctx.Err()
	}}
	p, requests, _ := offloadingPlugin(t, off, func(c *Config) {
		c.BatchSize = 100
		c.BatchFlushIntv = time.Hour
		// The uploads of a run end get half of this, 500ms.
		c.ShutdownTimeout = time.Second
		c.Logger = &recordingLogger{}
	})

	const runs = 4
	var waited [runs]time.Duration
	for i := range runs {
		mCtx := &mockInvocationContext{
			ctx:          context.Background(),
			agentName:    "test-agent",
			sessionID:    "sess-id",
			invocationID: fmt.Sprintf("inv-%d", i),
			userID:       "user-id",
		}
		if _, err := p.BeforeModelCallback()(mCtx, &model.LLMRequest{Contents: []*genai.Content{imageContent(1, 2, 3)}}); err != nil {
			t.Fatalf("BeforeModelCallback() error: %v", err)
		}
		start := time.Now()
		p.AfterRunCallback()(mCtx)
		waited[i] = time.Since(start)
	}

	if waited[0] < 250*time.Millisecond {
		t.Errorf("the first run end took %v, want it to wait about 500ms for the uploads", waited[0])
	}
	for i, d := range waited[1:] {
		if d >= 250*time.Millisecond {
			t.Errorf("run end %d took %v, want < 250ms: the uploads are paused once those of the first run end have timed out", i+1, d)
		}
	}
	if got := len(off.uploads()); got != 1 {
		t.Errorf("the offloader was called %d times, want once, by the first run end", got)
	}

	rows := collectRows(t, requests, func(rows []map[string]any) bool {
		return len(rowsOfType(rows, EventInvocationCompleted)) == runs
	})
	llmRows := rowsOfType(rows, EventLLMRequest)
	if len(llmRows) != runs {
		t.Fatalf("got %d LLM_REQUEST rows, want %d: no run end may lose rows", len(llmRows), runs)
	}
	for i, row := range llmRows {
		if got := contentPartsOf(t, row)[0]["text"]; got != uploadFailedText {
			t.Errorf("row %d part text = %v, want %q", i, got, uploadFailedText)
		}
	}
}

func TestGCSOffloading_RunEndPausesStalledUploads(t *testing.T) {
	for _, tt := range []struct {
		name           string
		stalledBatch   int
		partialSuccess bool
		uploadDelay    time.Duration
	}{
		{name: "partial outage in first batch", partialSuccess: true, uploadDelay: 100 * time.Millisecond},
		{name: "outage in later batch", stalledBatch: 1, uploadDelay: 100 * time.Millisecond},
		{name: "partial outage in later batch", stalledBatch: 1, partialSuccess: true, uploadDelay: 100 * time.Millisecond},
		{name: "outage after slow first batch", stalledBatch: 1, uploadDelay: 2800 * time.Millisecond},
		{name: "partial outage after slow first batch", stalledBatch: 1, partialSuccess: true, uploadDelay: 2800 * time.Millisecond},
	} {
		t.Run(tt.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				off := &fakeOffloader{hook: func(ctx context.Context, req GCSUploadRequest) error {
					if int(req.Data[0])/2 < tt.stalledBatch || (tt.partialSuccess && req.Data[0]%2 == 0) {
						time.Sleep(tt.uploadDelay)
						return nil
					}
					<-ctx.Done()
					return ctx.Err()
				}}
				cfg := DefaultConfig()
				cfg.BatchSize = 2
				cfg.Logger = &recordingLogger{}
				cfg.GCSOffloader = off
				bp, err := NewBatchProcessor(context.Background(), nil, "test_stream", cfg)
				if err != nil {
					t.Fatal(err)
				}
				defer bp.Close()
				stream := &mockStream{recvRes: &storagepb.AppendRowsResponse{}, sendDelay: 100 * time.Millisecond}
				bp.streamWriter.stream = stream
				opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 50, reserveBytes: bp.tryReserveUploadBytes}
				const rowCount = 16
				for i := range rowCount {
					row := queuedRow(opts, imageContent(byte(i)))
					row["event_type"] = EventLLMRequest
					bp.queue <- row
				}

				start := time.Now()
				bp.flushRunEnd()
				if elapsed := time.Since(start); elapsed > cfg.ShutdownTimeout {
					t.Errorf("run-end flush took %v, exceeds %v", elapsed, cfg.ShutdownTimeout)
				}
				if got, want := len(off.uploads()), 2*(tt.stalledBatch+1); got != want {
					t.Errorf("upload attempts = %d, want %d; later batches must skip stalled storage", got, want)
				}
				var rows []map[string]any
				for _, req := range stream.requests() {
					rows = append(rows, decodeArrowRows(t, req)...)
				}
				if len(rows) != rowCount {
					t.Fatalf("persisted rows = %d, want %d", len(rows), rowCount)
				}
				for i, row := range rows {
					wantMode := storageModeInline
					if i/2 < tt.stalledBatch || (i/2 == tt.stalledBatch && tt.partialSuccess && i%2 == 0) {
						wantMode = storageModeGCSReference
					}
					if contentPartsOf(t, row)[0]["storage_mode"] != wantMode {
						t.Errorf("row %d has incorrect storage mode, want %s", i, wantMode)
					}
				}
				if got := bp.pendingUploadBytes.Load(); got != 0 {
					t.Errorf("pending upload bytes = %d, want 0", got)
				}
			})
		})
	}
}

// TestGCSOffloading_RunEndOwnDeadlineErrorBeforeTheDeadlineDoesNotPause runs
// a run-end flush under the default 10s ShutdownTimeout whose first batch's
// BigQuery write takes 7s, so that the next batch starts with 3s left, enough
// for its deadline to pause the uploads, and has 1.5s for its two uploads.
// One upload fails after 1.2s, more than three quarters of that time, with a
// deadline error of its own, such as a custom offloader's per-request
// timeout, while that deadline has not passed yet; the other succeeds once
// the deadline has ended the uploads. The deadline cut no upload short, and
// an upload succeeded, so the uploads are not paused, and the next batch,
// which a later run-end flush writes, uploads its part.
func TestGCSOffloading_RunEndOwnDeadlineErrorBeforeTheDeadlineDoesNotPause(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		off := &fakeOffloader{hook: func(ctx context.Context, req GCSUploadRequest) error {
			switch req.Data[0] {
			case 1:
				time.Sleep(1200 * time.Millisecond)
				return context.DeadlineExceeded
			case 2:
				<-ctx.Done()
			}
			return nil
		}}
		cfg := DefaultConfig()
		cfg.BatchSize = 2
		cfg.Logger = &recordingLogger{}
		cfg.GCSOffloader = off
		bp, err := NewBatchProcessor(context.Background(), nil, "test_stream", cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer bp.Close()
		writes := 0
		stream := &mockStream{recvRes: &storagepb.AppendRowsResponse{}, onSend: func() {
			writes++
			if writes == 1 {
				time.Sleep(7 * time.Second)
			}
		}}
		bp.streamWriter.stream = stream
		opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 50, reserveBytes: bp.tryReserveUploadBytes}
		bp.queue <- map[string]any{"event_type": EventUserMessageReceived}
		bp.queue <- map[string]any{"event_type": EventUserMessageReceived}
		for i := byte(1); i <= 2; i++ {
			row := queuedRow(opts, imageContent(i))
			row["event_type"] = EventLLMRequest
			bp.queue <- row
		}
		bp.flushRunEnd()
		if got := len(off.uploads()); got != 2 {
			t.Fatalf("upload attempts = %d, want 2", got)
		}
		if bp.uploadCooldownUntil.Load() != 0 {
			t.Error("an upload that failed with a deadline error of its own before the deadline paused the uploads")
		}

		next := queuedRow(opts, imageContent(3))
		next["event_type"] = EventLLMRequest
		bp.queue <- next
		bp.flushRunEnd()
		if got := len(off.uploads()); got != 3 {
			t.Errorf("upload attempts after the next flush = %d, want 3", got)
		}
		if got := contentPartsOf(t, next)[0]["storage_mode"]; got != storageModeGCSReference {
			t.Errorf("next batch storage_mode = %v, want %s", got, storageModeGCSReference)
		}
		if got := bp.pendingUploadBytes.Load(); got != 0 {
			t.Errorf("pending upload bytes = %d, want 0", got)
		}
	})
}

// TestGCSOffloading_CloseDuringStorageStallKeepsRows checks that a Cloud
// Storage outage does not cost BigQuery rows on any flush path. The flush of a
// tick is stuck waiting for Cloud Storage while more rows queue up behind it,
// and Close has to upload (under a separate half-ShutdownTimeout upload budget)
// and write all of them within the ShutdownTimeout write budget, with BigQuery
// taking 150ms for each append.
func TestGCSOffloading_CloseDuringStorageStallKeepsRows(t *testing.T) {
	ctx := context.Background()
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	requests := make(chan *storagepb.AppendRowsRequest, 100)
	storagepb.RegisterBigQueryWriteServer(srv, &delayedBigQueryWriteServer{delay: 150 * time.Millisecond, requests: requests})
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatal(err)
	}
	bqClient, err := bq.NewClient(ctx, "p", option.WithHTTPClient(&http.Client{Transport: &mockTransport{
		roundTrip: func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader("{}"))}, nil
		},
	}}))
	if err != nil {
		t.Fatal(err)
	}

	config := DefaultConfig()
	config.ProjectID, config.DatasetID, config.TableName = "p", "d", "t"
	config.BatchSize = 10
	config.ShutdownTimeout = 2 * time.Second
	config.Logger = &recordingLogger{}
	// Cloud Storage hangs: every upload waits until its context is done.
	config.GCSOffloader = &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		<-ctx.Done()
		return ctx.Err()
	}}
	p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, config, bqClient, writeClient)
	if err != nil {
		t.Fatal(err)
	}
	mCtx := &mockInvocationContext{ctx: ctx, agentName: "a", sessionID: "s", invocationID: "i", userID: "u"}
	logImage := func() {
		req := &model.LLMRequest{Contents: []*genai.Content{imageContent(1, 2, 3)}}
		if _, err := p.BeforeModelCallback()(mCtx, req); err != nil {
			t.Fatal(err)
		}
	}

	// The first rows go out with the next tick, whose flush then waits on
	// Cloud Storage while 85 more rows queue up behind it.
	for range 5 {
		logImage()
	}
	time.Sleep(1300 * time.Millisecond)
	for range 85 {
		logImage()
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}

	written := 0
	for len(requests) > 0 {
		written += int((<-requests).GetArrowRows().GetRows().GetRowCount())
	}
	if written != 90 {
		t.Errorf("rows written = %d, want 90: a Cloud Storage stall must not cost BigQuery rows", written)
	}
}

func TestGCSOffloading_FailedUploadPreservesJSONPrefixInSummary(t *testing.T) {
	off := &fakeOffloader{hook: func(context.Context, GCSUploadRequest) error {
		return errors.New("boom")
	}}
	p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
		c.MaxInlineTextSize = 1024
		c.MaxContentLen = 200
		c.Logger = &recordingLogger{}
	})
	msg := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: `{"q":"x"}`},
			{Text: strings.Repeat("y", 40*1024)},
		},
	}
	if _, err := p.OnUserMessageCallback()(mCtx, msg); err != nil {
		t.Fatalf("OnUserMessageCallback() error: %v", err)
	}
	p.AfterRunCallback()(mCtx)

	rows := collectRows(t, requests, untilEventType(EventInvocationCompleted))
	userRow := onlyRowOfType(t, rows, EventUserMessageReceived)
	userContent := jsonObjectColumn(t, userRow, columnContent)
	summary, _ := userContent["text_summary"].(string)
	if !strings.HasPrefix(summary, `{"q":"x"} | yyy`) || strings.Contains(summary, unparseableJSONPlaceholder) {
		t.Errorf("text_summary after failed upload = %q, want prefix %q without %q", summary, `{"q":"x"} | yyy`, unparseableJSONPlaceholder)
	}
}

func TestGCSOffloading_PreservesBracketLedProse(t *testing.T) {
	t.Run("non-sensitive bracket-led prose is offloaded intact", func(t *testing.T) {
		off := &fakeOffloader{}
		p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
			c.MaxInlineTextSize = 200
		})
		prose := "[Chapter 1] " + strings.Repeat("ordinary prose without secrets ", 20)

		row := logRequest(t, p, requests, mCtx, &model.LLMRequest{Contents: []*genai.Content{textContent("user", prose)}})

		uploads := off.uploads()
		if len(uploads) != 1 {
			t.Fatalf("got %d uploads, want 1", len(uploads))
		}
		if got := string(uploads[0].Data); got != prose {
			t.Errorf("uploaded text = %q, want original bracket-led prose preserved", got)
		}
		part := contentPartsOf(t, row)[0]
		if part["storage_mode"] != "GCS_REFERENCE" {
			t.Errorf("storage_mode = %v, want GCS_REFERENCE", part["storage_mode"])
		}
		if got, _ := part["text"].(string); !strings.HasPrefix(got, "[Chapter 1] ") || strings.Contains(got, unparseableJSONPlaceholder) {
			t.Errorf("part text preview = %q, want prefix \"[Chapter 1] \" without %q", got, unparseableJSONPlaceholder)
		}
		if row["is_truncated"] != false {
			t.Errorf("is_truncated = %v, want false", row["is_truncated"])
		}
	})

	t.Run("bracket-led malformed JSON with credential marker still fails closed", func(t *testing.T) {
		off := &fakeOffloader{}
		p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
			c.MaxInlineTextSize = 200
		})
		malformed := `[{"api_key":"sk-live-secret-999",` + strings.Repeat("x", 300)

		row := logRequest(t, p, requests, mCtx, &model.LLMRequest{Contents: []*genai.Content{textContent("user", malformed)}})

		if got := contentPartsOf(t, row)[0]["text"]; got != unparseableJSONPlaceholder {
			t.Errorf("part text = %v, want %q when credential marker is present", got, unparseableJSONPlaceholder)
		}
		if row["is_truncated"] != true {
			t.Errorf("is_truncated = %v, want true when malformed JSON with credential fails closed", row["is_truncated"])
		}
	})
}

func TestGCSOffloading_BuiltInOffloaderErrorPreservesScrubbedDetail(t *testing.T) {
	fs := newFakeStorage(t)
	fs.failWith = http.StatusForbidden
	opts, _ := storageClientOptions(fs)
	logger := &recordingLogger{}
	p, requests, mCtx := setupTestPluginWithConfig(t, func(c *Config) {
		c.GCSBucketName = "test-bucket"
		c.GCSClientOptions = opts
		c.Logger = logger
	})

	row := logRequest(t, p, requests, mCtx, &model.LLMRequest{Contents: []*genai.Content{imageContent(1, 2, 3)}})
	if err := p.Close(); err != nil {
		t.Fatalf("Close() error: %v", err)
	}

	if got := contentPartsOf(t, row)[0]["text"]; got != uploadFailedText {
		t.Errorf("part text = %v, want %q", got, uploadFailedText)
	}
	if !logger.contains("failed to write the Cloud Storage object") || !logger.contains("403") {
		t.Errorf("expected built-in offloader transport/API detail in log, got:\n%s", logger)
	}
	if logger.contains("?") || logger.contains("uploadType=") {
		t.Errorf("built-in offloader log = %s, want URL query parameters stripped", logger)
	}

	// Verify ownedUploadError strips URI query strings and scrubs key-value credentials before logging.
	summary := safeUploadErrorSummary(ownedUploadError{
		err: errors.New("Post \"https://storage.googleapis.com/upload/storage/v1/b/bkt/o?uploadType=multipart&access_token=test-access-tok\": api_key=sk-secret-123 x509: unknown authority"),
	})
	if strings.Contains(summary, "?") || strings.Contains(summary, "uploadType") || strings.Contains(summary, "test-access-tok") || strings.Contains(summary, "sk-secret-123") {
		t.Errorf("safeUploadErrorSummary leaked query or secret: %q", summary)
	}
	if !strings.Contains(summary, "https://storage.googleapis.com/upload/storage/v1/b/bkt/o") || !strings.Contains(summary, "x509: unknown authority") || !strings.Contains(summary, redactedValue) {
		t.Errorf("safeUploadErrorSummary = %q, want queryless URL and scrubbed diagnostic reason preserved", summary)
	}
}

func TestResolveUploads_BatchTimeoutCooldown(t *testing.T) {
	var shouldBlock atomic.Bool
	shouldBlock.Store(true)
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		if shouldBlock.Load() {
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
	logger := &recordingLogger{}
	bp := &BatchProcessor{
		config: Config{
			GCSOffloader:            off,
			GCSMaxConcurrentUploads: 2,
			GCSUploadTimeout:        40 * time.Millisecond,
			Logger:                  logger,
		},
	}

	// Batch 1 times out on GCSUploadTimeout and arms the cooldown.
	row1 := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}, imageContent(1))
	bp.resolveUploads(context.Background(), false, []map[string]any{row1})
	if got := len(off.uploads()); got != 1 {
		t.Fatalf("after batch 1 got %d upload attempts, want 1", got)
	}

	// Batch 2 runs during the active cooldown: skips calling UploadContent and returns immediately.
	row2 := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}, imageContent(2))
	start := time.Now()
	bp.resolveUploads(context.Background(), false, []map[string]any{row2})
	if elapsed := time.Since(start); elapsed >= 20*time.Millisecond {
		t.Errorf("cooldown batch took %v, want < 20ms without waiting on offloader", elapsed)
	}
	if got := len(off.uploads()); got != 1 {
		t.Errorf("after cooldown batch got %d upload attempts, want 1 (upload skipped during cooldown)", got)
	}
	if got := row2[columnContentParts].([]map[string]any)[0]["text"]; got != uploadFailedText {
		t.Errorf("cooldown batch part text = %v, want %q", got, uploadFailedText)
	}
	if !logger.contains("upload skipped during timeout cooldown") {
		t.Errorf("expected cooldown log entry, got:\n%s", logger)
	}

	// Once cooldown expires and storage recovers, Batch 3 uploads normally.
	time.Sleep(50 * time.Millisecond)
	shouldBlock.Store(false)
	row3 := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}, imageContent(3))
	bp.resolveUploads(context.Background(), false, []map[string]any{row3})
	if got := len(off.uploads()); got != 2 {
		t.Errorf("after cooldown expiry got %d upload attempts, want 2", got)
	}
	if got := row3[columnContentParts].([]map[string]any)[0]["storage_mode"]; got != "GCS_REFERENCE" {
		t.Errorf("after cooldown expiry storage_mode = %v, want GCS_REFERENCE", got)
	}
}

// TestResolveUploads_DeadlinesPauseTheUploads checks which ways of ending the
// uploads of a batch show that Cloud Storage is stalled, and so make the
// batches after it fall back without calling the offloader, and which show
// nothing about Cloud Storage. It also checks which of them keep the part that
// they cut short on its row, for a retry if the batch is deferred. Neither the
// time that an upload is counted as having run, nor which deadline is counted
// as having ended the uploads, depends on how soon the offloader returns once
// its context is done.
func TestResolveUploads_DeadlinesPauseTheUploads(t *testing.T) {
	tests := []struct {
		name string
		// uploadTimeout is Config.GCSUploadTimeout.
		uploadTimeout time.Duration
		// pauseOnCallerDeadline reports whether a caller deadline is fresh and
		// allowed to pause uploads when it expires before uploadTimeout.
		pauseOnCallerDeadline bool
		// partialSuccess makes the first of two uploads in the batch succeed
		// before the second stalls until the deadline.
		partialSuccess bool
		// firstUploadTakes is how long that first upload takes to succeed,
		// and maxConcurrent is Config.GCSMaxConcurrentUploads.
		firstUploadTakes time.Duration
		maxConcurrent    int
		// uploads is how many uploads the first batch has, when that is not
		// one (or two with partialSuccess). When failAfter is set, an upload
		// that stalls fails with an error of its own after that long, unless
		// its context ends first. When returnLag is set, an upload that its
		// context ended returns that long after the context is done.
		uploads   int
		failAfter time.Duration
		returnLag time.Duration
		// newCtx returns the context that the first batch is resolved under.
		newCtx func(t *testing.T) context.Context
		// wantCalls (or [minCalls, maxCalls] when set) is how many uploads the
		// first batch starts, wantPaused whether the batch after it then skips
		// Cloud Storage, and wantKept whether the first batch keeps the part
		// that it cut short.
		wantCalls  int
		minCalls   int
		maxCalls   int
		wantPaused bool
		wantKept   bool
		// wantKeptCount overrides wantKept when a batch keeps more than one
		// part. firstUploadURI overrides the URI returned by the first upload
		// when it succeeds; firstUploadErr makes the first upload fail at once
		// with that error; afterDoneErr overrides the error returned once ctx
		// is done, and returnCause returns context.Cause(ctx) instead of
		// ctx.Err().
		wantKeptCount    int
		firstUploadURI   string
		firstUploadErr   error
		afterDoneErr     error
		returnCause      bool
		wantLogSubstring string
		// firstUploadErrAfter delays the failure of the first upload with
		// firstUploadErr by that long, during which its context is still
		// live, and succeedOnceDone makes an upload that waits for its
		// context to be done return success then.
		firstUploadErrAfter time.Duration
		succeedOnceDone     bool
	}{
		{
			name:          "the upload timeout runs out",
			uploadTimeout: 40 * time.Millisecond,
			newCtx:        func(*testing.T) context.Context { return context.Background() },
			wantCalls:     1,
			wantPaused:    true,
		},
		{
			name:                  "a fresh deadline of the caller runs out before the upload timeout",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: true,
			// No upload of the batch succeeds, so its deadline pauses the
			// uploads, whether or not the upload that it cut short had run
			// long enough to count as stalled; the deadline is long enough
			// to leave room for a late start.
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:  1,
			wantPaused: true,
		},
		{
			name:                  "a depleted deadline of the caller runs out before the upload timeout",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: false,
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:  1,
			wantPaused: false,
			wantKept:   true,
		},
		{
			name:                  "an upload succeeds before the deadline runs out on another",
			uploadTimeout:         40 * time.Millisecond,
			pauseOnCallerDeadline: true,
			partialSuccess:        true,
			newCtx:                func(*testing.T) context.Context { return context.Background() },
			wantCalls:             2,
			wantPaused:            true,
		},
		{
			name:                  "an upload succeeds before a fresh deadline of the caller runs out on another",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: true,
			partialSuccess:        true,
			// As above, the deadline leaves room for a late start.
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:  2,
			wantPaused: true,
		},
		{
			name:                  "a fresh deadline of the caller cuts short an upload that ran for 60% of it",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: true,
			partialSuccess:        true,
			// The first upload holds the only slot for 200ms of the 500ms
			// deadline, so the second runs for at most 300ms of it, less than
			// three quarters (375ms).
			firstUploadTakes: 200 * time.Millisecond,
			maxConcurrent:    1,
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:  2,
			wantPaused: false,
			wantKept:   true,
		},
		{
			name:                  "a fresh deadline of the caller cuts short an upload that ran for 85% of it",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: true,
			partialSuccess:        true,
			// The first upload holds the only slot for 180ms of the 1200ms
			// deadline, so the second runs for about 1020ms of it, more than
			// three quarters (900ms) while still below seven eighths (1050ms).
			firstUploadTakes: 180 * time.Millisecond,
			maxConcurrent:    1,
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 1200*time.Millisecond)
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:  2,
			wantPaused: true,
		},
		{
			name:                  "a fresh deadline of the caller cuts short an upload that ran for 60% of it, and the offloader returns late",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: true,
			partialSuccess:        true,
			// As in the 60% case, but the offloader returns 150ms after the
			// deadline, so the second upload takes 450ms from its start to its
			// return, more than three quarters (375ms) of the deadline. What
			// counts is the 300ms that it ran before the deadline.
			firstUploadTakes: 200 * time.Millisecond,
			maxConcurrent:    1,
			returnLag:        150 * time.Millisecond,
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:  2,
			wantPaused: false,
			wantKept:   true,
		},
		{
			name:                  "uploads fail with errors until a fresh deadline of the caller runs out",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: true,
			// Each upload fails after 200ms, well short of three quarters
			// (750ms) of the 1s deadline, so the deadline cuts uploads short
			// with none having stalled and none having succeeded.
			uploads:       7,
			failAfter:     200 * time.Millisecond,
			maxConcurrent: 1,
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), time.Second)
				t.Cleanup(cancel)
				return ctx
			},
			minCalls:   2,
			maxCalls:   6,
			wantPaused: true,
		},
		{
			name:          "the upload timeout runs out before a deadline of the caller, and the offloader returns after both",
			uploadTimeout: 300 * time.Millisecond,
			// The upload timeout ends the stalled upload 200ms before the
			// 500ms deadline, but the offloader returns 300ms later, 100ms
			// after the deadline. It is still the upload timeout that ended
			// the upload: it pauses the uploads, and the upload is not kept.
			returnLag: 300 * time.Millisecond,
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:  1,
			wantPaused: true,
		},
		{
			name:                  "the upload timeout runs out before a deadline of the caller, and an upload that succeeds returns after both",
			uploadTimeout:         100 * time.Millisecond,
			pauseOnCallerDeadline: true,
			partialSuccess:        true,
			// The first upload takes 800ms to succeed, whatever its context
			// says, so the second one does not start. The upload timeout ended
			// the uploads 500ms before the 600ms deadline, so the second
			// upload is not kept for a retry, though the deadline has passed
			// by the time that the first upload returns.
			firstUploadTakes: 800 * time.Millisecond,
			maxConcurrent:    1,
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 600*time.Millisecond)
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:  1,
			wantPaused: false,
		},
		{
			name:                  "the caller cancels the context",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: true,
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithCancel(context.Background())
				t.Cleanup(cancel)
				time.AfterFunc(40*time.Millisecond, cancel)
				return ctx
			},
			wantCalls:  1,
			wantPaused: false,
			wantKept:   true,
		},
		{
			name:                  "the context is canceled on arrival",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: true,
			newCtx: func(*testing.T) context.Context {
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx
			},
			wantCalls:  0,
			wantPaused: false,
			wantKept:   true,
		},
		{
			name:                  "the deadline of the context has passed on arrival",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: true,
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:  0,
			wantPaused: false,
			wantKept:   true,
		},
		{
			name:                  "an offloader returns a non-context error once a depleted deadline of the caller runs out",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: false,
			afterDoneErr:          errors.New("opaque transport error"),
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:  1,
			wantPaused: false,
			wantKept:   true,
		},
		{
			name:                  "an upload fails early with a non-context error and another is cut short by a depleted deadline",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: false,
			uploads:               2,
			firstUploadErr:        errors.New("early failure"),
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:     2,
			wantPaused:    false,
			wantKeptCount: 2,
		},
		{
			name:             "the upload timeout runs out and the offloader returns the context cause",
			uploadTimeout:    100 * time.Millisecond,
			returnCause:      true,
			newCtx:           func(*testing.T) context.Context { return context.Background() },
			wantCalls:        1,
			wantPaused:       true,
			wantLogSubstring: "context deadline exceeded; keeping the part inline",
		},
		{
			name:                  "depleted caller deadline drops succeeded non-gs URI upload and keeps cut-short upload",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: false,
			partialSuccess:        true,
			uploads:               2,
			firstUploadURI:        "https://example.com/not-gs",
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:     2,
			wantPaused:    false,
			wantKept:      true,
			wantKeptCount: 1,
		},
		{
			name:                  "live caller context does not keep upload when offloader returns immediate DeadlineExceeded",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: false,
			uploads:               1,
			firstUploadErr:        context.DeadlineExceeded,
			newCtx:                func(*testing.T) context.Context { return context.Background() },
			wantCalls:             1,
			wantPaused:            false,
			wantKept:              false,
			wantKeptCount:         0,
		},
		{
			name:                  "an upload fails with a deadline error of its own before a fresh deadline of the caller, and another succeeds once that deadline ends the uploads",
			uploadTimeout:         time.Minute,
			pauseOnCallerDeadline: true,
			// The first upload fails after 800ms of the 1s deadline, more than
			// three quarters (750ms) of it, but with a deadline error of its
			// own, such as a per-request timeout, while its context is live:
			// the deadline did not cut it short. The second succeeds once the
			// deadline has passed, so the deadline ended the uploads and an
			// upload succeeded, which pauses nothing. The failed upload is kept
			// for a retry, as any upload that returned an error once the
			// deadline ended the uploads is.
			uploads:             2,
			firstUploadErr:      context.DeadlineExceeded,
			firstUploadErrAfter: 800 * time.Millisecond,
			succeedOnceDone:     true,
			newCtx: func(t *testing.T) context.Context {
				ctx, cancel := context.WithTimeout(context.Background(), time.Second)
				t.Cleanup(cancel)
				return ctx
			},
			wantCalls:  2,
			wantPaused: false,
			wantKept:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var stalled atomic.Bool
			var callCount atomic.Int32
			var firstCallPath atomic.Pointer[string]
			stalled.Store(true)
			off := &fakeOffloader{
				hook: func(ctx context.Context, req GCSUploadRequest) error {
					if !stalled.Load() {
						return nil
					}
					call := callCount.Add(1)
					if call == 1 {
						path := req.Path
						firstCallPath.Store(&path)
					}
					if tt.partialSuccess && call == 1 {
						time.Sleep(tt.firstUploadTakes)
						return nil
					}
					if tt.firstUploadErr != nil && call == 1 {
						time.Sleep(tt.firstUploadErrAfter)
						return tt.firstUploadErr
					}
					if tt.failAfter > 0 {
						select {
						case <-time.After(tt.failAfter):
							return errors.New("storage unavailable")
						case <-ctx.Done():
							return ctx.Err()
						}
					}
					<-ctx.Done()
					time.Sleep(tt.returnLag)
					if tt.returnCause {
						return context.Cause(ctx)
					}
					if tt.afterDoneErr != nil {
						return tt.afterDoneErr
					}
					if tt.succeedOnceDone {
						return nil
					}
					return ctx.Err()
				},
				uri: func(req GCSUploadRequest) string {
					if p := firstCallPath.Load(); stalled.Load() && tt.firstUploadURI != "" && p != nil && req.Path == *p {
						return tt.firstUploadURI
					}
					return "gs://fake-bucket/" + req.Path
				},
			}
			logger := &recordingLogger{}
			bp := &BatchProcessor{config: Config{GCSOffloader: off, GCSUploadTimeout: tt.uploadTimeout, GCSMaxConcurrentUploads: tt.maxConcurrent, Logger: logger}}
			opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}

			firstContents := []*genai.Content{imageContent(1)}
			if tt.partialSuccess {
				firstContents = []*genai.Content{imageContent(1), imageContent(2)}
			}
			if tt.uploads > 0 {
				firstContents = nil
				for range tt.uploads {
					firstContents = append(firstContents, imageContent(1))
				}
			}
			first := queuedRow(opts, firstContents...)
			bp.resolveUploads(tt.newCtx(t), tt.pauseOnCallerDeadline, []map[string]any{first})
			firstCalls := len(off.uploads())
			minCalls, maxCalls := tt.minCalls, tt.maxCalls
			if minCalls == 0 && maxCalls == 0 {
				minCalls, maxCalls = tt.wantCalls, tt.wantCalls
			}
			if firstCalls < minCalls || firstCalls > maxCalls {
				t.Fatalf("the first batch started %d uploads, want [%d, %d]", firstCalls, minCalls, maxCalls)
			}
			// A batch that keeps cut-short parts keeps wantKeptCount when set,
			// or one part when wantKept is true.
			wantKept := tt.wantKeptCount
			if wantKept == 0 && tt.wantKept {
				wantKept = 1
			}
			var kept int
			if d, ok := first[deferredUploadsKey].(*deferredRowUploads); ok {
				kept = len(d.pending)
			}
			if kept != wantKept {
				t.Errorf("the first batch kept %d parts for a retry, want %d", kept, wantKept)
			}
			if !tt.wantKept && tt.wantKeptCount == 0 {
				if _, hasKept := first[deferredUploadsKey]; hasKept {
					t.Fatalf("expected no deferredUploadsKey on row when wantKept=false")
				}
			}
			if tt.wantLogSubstring != "" {
				gotLog := logger.String()
				if !strings.Contains(gotLog, tt.wantLogSubstring) {
					t.Errorf("log = %q, want substring %q", gotLog, tt.wantLogSubstring)
				}
				if strings.Contains(gotLog, "error type") {
					t.Errorf("log = %q, must not fall back to generic error type summary", gotLog)
				}
			}

			// Cloud Storage is back by the time that the next batch is
			// resolved: whether that batch calls it depends on the pause alone.
			stalled.Store(false)
			next := queuedRow(opts, imageContent(3))
			bp.resolveUploads(context.Background(), false, []map[string]any{next})

			part := contentPartsOf(t, next)[0]
			wantCalls, wantMode, wantText := firstCalls+1, "GCS_REFERENCE", "[MEDIA OFFLOADED]"
			if tt.wantPaused {
				wantCalls, wantMode, wantText = firstCalls, "INLINE", uploadFailedText
			}
			if got := len(off.uploads()); got != wantCalls {
				t.Errorf("the offloader was called %d times in all, want %d", got, wantCalls)
			}
			if part["storage_mode"] != wantMode || part["text"] != wantText {
				t.Errorf("the next batch has storage_mode %v and text %v, want %s and %q", part["storage_mode"], part["text"], wantMode, wantText)
			}
		})
	}
}

func TestGCSOffloading_ConfigurablePendingUploadBytesCap(t *testing.T) {
	bp := &BatchProcessor{
		config: applyDefaults(Config{
			GCSOffloader:             &fakeOffloader{},
			GCSMaxPendingUploadBytes: 80,
			Logger:                   &recordingLogger{},
		}),
		queue: make(chan map[string]any, 4),
	}
	p := newAddContentPlugin(t, func(c *Config) {
		c.GCSOffloader = bp.config.GCSOffloader
		c.GCSMaxPendingUploadBytes = 80
	})
	p.processor = bp

	rowTooLarge := map[string]any{}
	p.addContent(rowTooLarge, EventLLMRequest, &eventData{content: &model.LLMRequest{Contents: []*genai.Content{imageContent(make([]byte, 100)...)}}})
	if _, ok := rowTooLarge[rawPartsKey]; ok {
		t.Fatal("100-byte payload should be rejected when GCSMaxPendingUploadBytes is 80")
	}

	rowFits := map[string]any{}
	p.addContent(rowFits, EventLLMRequest, &eventData{content: &model.LLMRequest{Contents: []*genai.Content{imageContent(make([]byte, 60)...)}}})
	if _, ok := rowFits[rawPartsKey]; !ok {
		t.Fatal("60-byte payload should be reserved when GCSMaxPendingUploadBytes is 80")
	}
	if got := bp.pendingUploadBytes.Load(); got != 60 {
		t.Errorf("pendingUploadBytes = %d, want 60", got)
	}
}

type panicOnBudgetLogger struct {
	recordingLogger
}

func (l *panicOnBudgetLogger) Printf(format string, v ...any) {
	if strings.Contains(format, "upload byte budget") {
		panic("logger panic after first part reserved bytes")
	}
	l.recordingLogger.Printf(format, v...)
}

func TestGCSOffloading_BuildRowPanicReleasesReservedBytes(t *testing.T) {
	logger := &panicOnBudgetLogger{}
	bp := &BatchProcessor{
		config: applyDefaults(Config{
			GCSOffloader:             &fakeOffloader{},
			GCSMaxPendingUploadBytes: 100,
			Logger:                   logger,
		}),
		queue: make(chan map[string]any, 4),
	}
	p := newAddContentPlugin(t, func(c *Config) {
		c.GCSOffloader = bp.config.GCSOffloader
		c.GCSMaxPendingUploadBytes = 100
		c.Logger = logger
	})
	p.processor = bp

	// Part 0 reserves 60 bytes (<= 100). Part 1 (60 bytes) exceeds the 100-byte
	// cap and logs a budget warning whose Logger.Printf panics before Append;
	// addContent's defer attaches the reserved part to row[rawPartsKey] so
	// buildRow's defer releases it.
	p.writeEvent(context.Background(), identity{}, EventLLMRequest, rowSpan{}, func(d *eventData) {
		d.content = &model.LLMRequest{Contents: []*genai.Content{{
			Role: "user",
			Parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: make([]byte, 60)}},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: make([]byte, 60)}},
			},
		}}}
	})

	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes after buildRow panic = %d, want 0 (reservation must be released)", got)
	}
}

func TestGCSOffloading_RunEndDeadlineDoesNotPauseHealthyStorage(t *testing.T) {
	ctx := context.Background()
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	requests := make(chan *storagepb.AppendRowsRequest, 100)
	// BigQuery AppendRows takes 1.5s on the first batch.
	storagepb.RegisterBigQueryWriteServer(srv, &delayedBigQueryWriteServer{
		delay:    1500 * time.Millisecond,
		requests: requests,
	})
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = writeClient.Close() })

	logger := &recordingLogger{}
	// Cloud Storage is healthy: every upload succeeds in 300ms (well below the 10s GCSUploadTimeout).
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		select {
		case <-time.After(300 * time.Millisecond):
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}}

	cfg := DefaultConfig()
	cfg.ProjectID, cfg.DatasetID, cfg.TableName = "p", "d", "t"
	cfg.BatchSize = 2
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 2 * time.Second
	cfg.Logger = logger
	cfg.GCSOffloader = off

	bp, err := NewBatchProcessor(ctx, writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	bp.Start()

	enqueueImageRow := func(id string) {
		row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3))
		row["event_type"] = EventLLMRequest
		row["session_id"] = id
		bp.queue <- row
	}

	// First run: 3 image rows + 1 INVOCATION_COMPLETED = 2 batches of 2.
	// Enqueue directly on bp.queue so no background flush goroutine races
	// with flushRunEnd for flushMu. Batch 1 uses ~1.8s (300ms upload + 1.5s
	// BigQuery write), leaving ~200ms of the 2s run-end flush budget for
	// Batch 2's 300ms upload (capped at ~100ms).
	for i := range 3 {
		enqueueImageRow(fmt.Sprintf("run1-%d", i))
	}
	bp.queue <- map[string]any{"event_type": EventInvocationCompleted, "session_id": "run1-done"}
	bp.flushRunEnd()
	if bp.uploadCooldownUntil.Load() != 0 {
		t.Fatalf("uploadCooldownUntil = %d after depleted second-batch run-end deadline, want 0", bp.uploadCooldownUntil.Load())
	}

	// Second run: storage is still healthy (300ms << 10s GCSUploadTimeout).
	// Both the deferred Batch 2 from the first run and the new image row
	// must be uploaded to GCS, not skipped with "upload skipped during timeout cooldown".
	enqueueImageRow("run2-0")
	bp.queue <- map[string]any{"event_type": EventInvocationCompleted, "session_id": "run2-done"}
	bp.flushRunEnd()
	bp.Close()

	if logger.contains("upload skipped during timeout cooldown") {
		t.Fatalf("healthy storage was paused after run-end flush deadline expired:\n%s", logger)
	}
	rows := collectRows(t, requests, func(rows []map[string]any) bool {
		return len(rowsOfType(rows, EventLLMRequest)) >= 4
	})
	llmRows := rowsOfType(rows, EventLLMRequest)
	if len(llmRows) < 4 {
		t.Fatalf("got %d LLM_REQUEST rows, want >= 4", len(llmRows))
	}
	for i, r := range llmRows {
		part := contentPartsOf(t, r)[0]
		if part["storage_mode"] != "GCS_REFERENCE" {
			t.Errorf("LLM_REQUEST row %d (session_id=%v) storage_mode = %v (text=%v), want GCS_REFERENCE", i, r["session_id"], part["storage_mode"], part["text"])
		}
	}
}

func TestGCSOffloading_RunEndContinuousArrivalsDoNotPauseHealthyStorage(t *testing.T) {
	ctx := context.Background()
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	requests := make(chan *storagepb.AppendRowsRequest, 200)
	storagepb.RegisterBigQueryWriteServer(srv, &delayedBigQueryWriteServer{
		delay:    100 * time.Millisecond,
		requests: requests,
	})
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = writeClient.Close() })

	logger := &recordingLogger{}
	firstUploadStarted := make(chan struct{})
	var firstUploadOnce sync.Once
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		firstUploadOnce.Do(func() { close(firstUploadStarted) })
		select {
		case <-time.After(60 * time.Millisecond):
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}}

	cfg := DefaultConfig()
	cfg.ProjectID, cfg.DatasetID, cfg.TableName = "p", "d", "t"
	cfg.BatchSize = 2
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = time.Second
	cfg.Logger = logger
	cfg.GCSOffloader = off

	bp, err := NewBatchProcessor(ctx, writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	bp.Start()

	newImageRow := func() map[string]any {
		row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3))
		row["event_type"] = EventLLMRequest
		return row
	}

	// Enqueue the first batch directly onto bp.queue before starting flushRunEnd
	// so flushRunEnd deterministically acquires flushMu first, then start the
	// continuous feeder once the first upload is in flight under flushRunEnd.
	bp.queue <- newImageRow()
	bp.queue <- newImageRow()

	stopFeeder := make(chan struct{})
	var feederWG sync.WaitGroup
	feederWG.Add(1)
	go func() {
		defer feederWG.Done()
		select {
		case <-firstUploadStarted:
		case <-stopFeeder:
			return
		}
		ticker := time.NewTicker(20 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-stopFeeder:
				return
			case <-ticker.C:
				bp.Append(newImageRow())
			}
		}
	}()

	bp.flushRunEnd()
	close(stopFeeder)
	feederWG.Wait()

	callsBeforeRun2 := len(off.uploads())
	for range 10 {
		bp.queue <- newImageRow()
	}
	bp.flushRunEnd()
	bp.Close()

	if logger.contains("upload skipped during timeout cooldown") {
		t.Fatalf("healthy storage was paused after continuous arrivals during run-end flush:\n%s", logger)
	}
	if callsInRun2 := len(off.uploads()) - callsBeforeRun2; callsInRun2 < 10 {
		t.Errorf("offloader calls after first run-end flush = %d, want >= 10 (storage should not be paused)", callsInRun2)
	}
}

func TestGCSOffloading_FlushRunEndCapsSecondBatchUploadAtHalfRemainingBudget(t *testing.T) {
	ctx := context.Background()
	requests := make(chan *storagepb.AppendRowsRequest, 16)
	// serverDelay stays 0 for the warm-up write below.
	var serverDelay atomic.Int64
	writeClient := newScriptedWriteClient(t, &dynamicDelayBigQueryWriteServer{
		delay:    &serverDelay,
		requests: requests,
	})

	// Only the second batch has an upload, and that upload stalls until its context expires.
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		<-ctx.Done()
		return ctx.Err()
	}}
	logger := &recordingLogger{}
	cfg := DefaultConfig()
	cfg.BatchSize = 1
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 2 * time.Second
	cfg.RetryConfig.MaxRetries = 0
	cfg.GCSOffloader = off
	cfg.Logger = logger

	bp, err := NewBatchProcessor(ctx, writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	t.Cleanup(bp.Close)

	// Open the connection and the write stream first, so that setting them up
	// does not use up the run-end budget below.
	warmCtx, cancelWarm := context.WithTimeout(ctx, 5*time.Second)
	defer cancelWarm()
	if err := bp.writeBatch(warmCtx, []map[string]any{{"event_type": EventUserMessageReceived}}); err != nil {
		t.Fatalf("warm-up writeBatch: %v", err)
	}
	<-requests
	serverDelay.Store(int64(time.Second))

	// Batch 0: no upload, takes 1s in AppendRows, leaving ~1s of the 2s ShutdownTimeout.
	bp.queue <- map[string]any{"event_type": EventUserMessageReceived}
	// Batch 1: has an image upload; its uploadCtx must be capped at half the remaining ~1s (~500ms),
	// not half of the initial ShutdownTimeout (1s).
	row2 := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3))
	row2["event_type"] = EventLLMRequest
	bp.queue <- row2

	bp.flushRunEnd()

	uploads := off.uploads()
	if len(uploads) != 1 {
		t.Fatalf("got %d upload attempts, want 1", len(uploads))
	}
	if !uploads[0].HasDeadline || uploads[0].Remaining < 200*time.Millisecond || uploads[0].Remaining > 700*time.Millisecond {
		t.Errorf("second batch upload context remaining = %v (hasDeadline=%v), want in [200ms, 700ms] (half of remaining ~1s budget)", uploads[0].Remaining, uploads[0].HasDeadline)
	}
	if bp.uploadCooldownUntil.Load() == 0 {
		t.Error("uploadCooldownUntil = 0, want non-zero (the second batch started with more than a quarter of ShutdownTimeout left, and its upload stalled for all of the batch's upload time)")
	}
}

func TestGCSOffloading_FlushRunEndPausesBatchStartingWithThirtyPercentBudget(t *testing.T) {
	ctx := context.Background()
	requests := make(chan *storagepb.AppendRowsRequest, 16)
	// serverDelay stays 0 for the warm-up write below (request 1).
	var serverDelay atomic.Int64
	writeClient := newScriptedWriteClient(t, &dynamicDelayBigQueryWriteServer{
		delay:    &serverDelay,
		requests: requests,
	})

	var stalled atomic.Bool
	stalled.Store(true)
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		// Batch 0's 1.4s BigQuery write (request 2) has already completed;
		// clear the server delay so Batch 1's inline fallback row (request 3)
		// writes within what remains of ShutdownTimeout.
		serverDelay.Store(0)
		if stalled.Load() {
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
	logger := &recordingLogger{}
	cfg := DefaultConfig()
	cfg.BatchSize = 1
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 2 * time.Second
	cfg.RetryConfig.MaxRetries = 0
	cfg.GCSOffloader = off
	cfg.Logger = logger

	bp, err := NewBatchProcessor(ctx, writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	t.Cleanup(bp.Close)

	// Open the connection and the write stream first (request 1, 0 delay), so
	// setting them up does not consume the run-end budget below.
	warmCtx, cancelWarm := context.WithTimeout(ctx, 5*time.Second)
	defer cancelWarm()
	if err := bp.writeBatch(warmCtx, []map[string]any{{"event_type": EventUserMessageReceived}}); err != nil {
		t.Fatalf("warm-up writeBatch: %v", err)
	}
	<-requests
	serverDelay.Store(int64(1400 * time.Millisecond))

	opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 32 * 1024, reserveBytes: bp.tryReserveUploadBytes}
	// Batch 0: inline text below the 32 KiB offload threshold (no uploads);
	// request 2 sleeps 1.4s, leaving ~600ms (~30% of the 2s ShutdownTimeout,
	// above 1/4 = 500ms and below 3/8 = 750ms) when Batch 1 starts.
	row0 := queuedRow(opts, &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "batch-0-inline"}}})
	row0["event_type"] = EventUserMessageReceived
	row0[eventIDField] = "batch-0-inline"
	bp.queue <- row0

	// Batch 1: 40 KiB upload stalling for its ~300ms half-remaining window.
	row1 := queuedRow(opts, &genai.Content{Role: "user", Parts: []*genai.Part{{Text: strings.Repeat("b", 40*1024)}}})
	row1["event_type"] = EventLLMRequest
	row1[eventIDField] = "batch-1-upload"
	bp.queue <- row1

	bp.flushRunEnd()

	if got := len(off.uploads()); got != 1 {
		t.Fatalf("got %d upload attempts, want 1", got)
	}
	if bp.uploadCooldownUntil.Load() == 0 {
		t.Error("uploadCooldownUntil = 0, want non-zero (Batch 1 started with ~30% of ShutdownTimeout left, above the 1/4 threshold, and its upload stalled)")
	}
	if got := len(bp.deferredBatch); got != 0 {
		t.Fatalf("deferredBatch length = %d, want 0", got)
	}
	for i, wantID := range []string{"batch-0-inline", "batch-1-upload"} {
		select {
		case req := <-requests:
			rows := decodeArrowRows(t, req)
			if len(rows) != 1 || rows[0][eventIDField] != wantID {
				summaries := make([]string, len(rows))
				for j, row := range rows {
					parts, _ := row[columnContentParts].([]map[string]any)
					summaries[j] = fmt.Sprintf("{event_id:%q parts:%d}", fmt.Sprint(row[eventIDField]), len(parts))
				}
				t.Fatalf("request %d got %d rows %v, want 1 row with event_id %q", i+2, len(rows), summaries, wantID)
			}
			if i == 1 {
				if mode := contentPartsOf(t, rows[0])[0]["storage_mode"]; mode != "INLINE" {
					t.Errorf("request 3 part storage_mode = %v, want INLINE", mode)
				}
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for request %d (%s)", i+2, wantID)
		}
	}
}

func TestGCSOffloading_FlushRunEndBatchAfterEmptyBatchPausesStalledStorage(t *testing.T) {
	ctx := context.Background()
	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	requests := make(chan *storagepb.AppendRowsRequest, 16)
	storagepb.RegisterBigQueryWriteServer(srv, &delayedBigQueryWriteServer{
		delay:    10 * time.Millisecond,
		requests: requests,
	})
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = writeClient.Close() })

	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		<-ctx.Done()
		return ctx.Err()
	}}
	logger := &recordingLogger{}
	cfg := DefaultConfig()
	cfg.BatchSize = 1
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = time.Second
	cfg.RetryConfig.MaxRetries = 0
	cfg.GCSOffloader = off
	cfg.Logger = logger

	bp, err := NewBatchProcessor(ctx, writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	bp.Start()
	defer bp.Close()

	// Batch 0 has no uploads and completes AppendRows in 10ms (~990ms of 1s remaining).
	bp.queue <- map[string]any{"event_type": EventUserMessageReceived}
	// Batch 1 starts with more than a quarter of the run-end budget left; when
	// storage stalls until Batch 1's ~495ms upload deadline, cooldown must be armed.
	row2 := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3))
	row2["event_type"] = EventLLMRequest
	bp.queue <- row2

	bp.flushRunEnd()

	if bp.uploadCooldownUntil.Load() == 0 {
		t.Fatalf("uploadCooldownUntil = 0, want non-zero when storage stalls for all of the upload time of a batch that starts with more than a quarter of the run-end budget left")
	}
}

func probeRow(t *testing.T, contents ...*genai.Content) map[string]any {
	t.Helper()
	return probeRowWithOptions(t, partsOptions{maxLen: 50, offload: true, inlineLimit: 10}, contents...)
}

func probeRowWithOptions(t *testing.T, opts partsOptions, contents ...*genai.Content) map[string]any {
	t.Helper()
	parts := &partsCollector{opts: opts}
	req := &model.LLMRequest{Contents: contents}
	content, contentTruncated, builder := buildLLMRequestContent(req, opts.maxLen, false, false, parts)
	b, err := json.Marshal(content)
	if err != nil {
		t.Fatal(err)
	}
	row := map[string]any{
		timestampField:     time.Date(2026, 3, 4, 5, 6, 7, 0, time.UTC),
		eventIDField:       "event1",
		"trace_id":         "trace1",
		"span_id":          "span1",
		columnContent:      string(b),
		columnContentParts: parts.parts,
		"is_truncated":     contentTruncated || parts.truncated,
	}
	if len(parts.pending) > 0 {
		row[rawPartsKey] = parts.pending
		if builder != nil && parts.hasPendingText() {
			builder.otherTruncated = parts.nonUploadTruncated
			row[summaryBuilderKey] = builder
		}
	}
	return row
}

// TestGCSOffloading_DeferredTextRetryTruncationFlag verifies that when a long
// text part (fallbackTruncated) is cut short by a depleted caller deadline,
// deferred via requeueDeferredUploads, and then uploads successfully on retry,
// the row's final is_truncated and content match a one-pass successful upload.
func TestGCSOffloading_DeferredTextRetryTruncationFlag(t *testing.T) {
	var stalled atomic.Bool
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		if stalled.Load() {
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
	bp := &BatchProcessor{config: Config{GCSOffloader: off, GCSUploadTimeout: time.Minute, Logger: &recordingLogger{}}}
	long := strings.Repeat("x", 200)

	control := probeRow(t, textContent("user", long))
	bp.resolveUploads(context.Background(), false, []map[string]any{control})

	probe := probeRow(t, textContent("user", long))
	stalled.Store(true)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	bp.resolveUploads(ctx, false, []map[string]any{probe})
	cancel()
	bp.requeueDeferredUploads([]map[string]any{probe})
	if _, ok := probe[rawPartsKey]; !ok {
		t.Fatalf("probe row was not requeued for upload")
	}
	stalled.Store(false)
	bp.resolveUploads(context.Background(), false, []map[string]any{probe})

	if control["is_truncated"] != probe["is_truncated"] {
		t.Errorf("is_truncated after deferred retry = %v, one-pass success = %v", probe["is_truncated"], control["is_truncated"])
	}
	if control[columnContent] != probe[columnContent] {
		t.Errorf("content after deferred retry differs:\n probe=%v\n control=%v", probe[columnContent], control[columnContent])
	}
}

// TestGCSOffloading_BinaryOnlyDeferralKeepsFailedTextSummary verifies that when
// a row has both a long text part (whose upload fails permanently with a
// non-deadline error) and a binary part (whose upload is cut short by a
// depleted caller deadline and deferred), retrying the binary upload preserves
// the failed text part's fallback summary in content and is_truncated.
func TestGCSOffloading_BinaryOnlyDeferralKeepsFailedTextSummary(t *testing.T) {
	var stalled atomic.Bool
	off := &fakeOffloader{hook: func(ctx context.Context, req GCSUploadRequest) error {
		if req.ContentType == textUploadContentType {
			return errors.New("permanent text upload failure")
		}
		if stalled.Load() {
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
	bp := &BatchProcessor{config: Config{GCSOffloader: off, GCSUploadTimeout: time.Minute, Logger: &recordingLogger{}}}
	long := strings.Repeat("y", 200)
	mk := func() map[string]any {
		return probeRow(t, &genai.Content{Role: "user", Parts: []*genai.Part{
			{Text: long},
			{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}}},
		}})
	}

	control := mk()
	bp.resolveUploads(context.Background(), false, []map[string]any{control})

	probe := mk()
	stalled.Store(true)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	bp.resolveUploads(ctx, false, []map[string]any{probe})
	cancel()
	bp.requeueDeferredUploads([]map[string]any{probe})
	if _, ok := probe[rawPartsKey]; !ok {
		t.Fatalf("probe row was not requeued for upload")
	}
	stalled.Store(false)
	bp.resolveUploads(context.Background(), false, []map[string]any{probe})

	if control["is_truncated"] != probe["is_truncated"] {
		t.Errorf("is_truncated after deferred retry = %v, one-pass = %v", probe["is_truncated"], control["is_truncated"])
	}
	if control[columnContent] != probe[columnContent] {
		t.Errorf("content after deferred retry differs:\n probe=%v\n control=%v", probe[columnContent], control[columnContent])
	}
}

func TestResolveUploads_PartialOutageBrownoutPausesUploads(t *testing.T) {
	var callNum atomic.Int32
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		// Every odd upload succeeds immediately; every even upload hangs until
		// the batch's GCSUploadTimeout expires (partial Cloud Storage brownout).
		if callNum.Add(1)%2 == 1 {
			return nil
		}
		<-ctx.Done()
		return ctx.Err()
	}}
	logger := &recordingLogger{}
	bp := &BatchProcessor{
		config: Config{
			GCSOffloader:            off,
			GCSMaxConcurrentUploads: 4,
			GCSUploadTimeout:        60 * time.Millisecond,
			Logger:                  logger,
		},
	}
	opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}

	// Batch 0 experiences a partial outage (upload 1 succeeds, upload 2 hangs
	// until GCSUploadTimeout) and must arm the cooldown.
	batch0 := []map[string]any{queuedRow(opts, imageContent(1), imageContent(2))}
	bp.resolveUploads(context.Background(), false, batch0)
	if bp.uploadCooldownUntil.Load() == 0 {
		t.Fatalf("uploadCooldownUntil = 0 after partial-outage batch timeout, want cooldown armed")
	}
	if got := len(off.uploads()); got != 2 {
		t.Fatalf("offloader calls after batch 0 = %d, want 2", got)
	}

	// Batches 1..3 run while cooldown is active and must return immediately
	// without waiting for GCSUploadTimeout on each batch.
	start := time.Now()
	for i := 1; i < 4; i++ {
		batch := []map[string]any{queuedRow(opts, imageContent(byte(2*i+1)), imageContent(byte(2*i+2)))}
		bp.resolveUploads(context.Background(), false, batch)
	}
	if elapsed := time.Since(start); elapsed >= 30*time.Millisecond {
		t.Errorf("batches 1..3 during brownout cooldown took %v, want < 30ms", elapsed)
	}
	if got := len(off.uploads()); got != 2 {
		t.Errorf("total offloader calls across 4 batches = %d, want 2 (batches 1..3 skipped during cooldown)", got)
	}
}

func TestGCSOffloading_DeferredUploadsHoldPendingByteReservation(t *testing.T) {
	var stalled atomic.Bool
	stalled.Store(true)
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		if stalled.Load() {
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
	logger := &recordingLogger{}
	bp := &BatchProcessor{
		config: applyDefaults(Config{
			GCSOffloader:             off,
			GCSUploadTimeout:         time.Minute,
			GCSMaxPendingUploadBytes: 64,
			Logger:                   logger,
		}),
		queue:                 make(chan map[string]any, 4),
		maxPendingUploadBytes: 64,
	}

	opts := partsOptions{
		maxLen:       1000,
		offload:      true,
		inlineLimit:  10,
		reserveBytes: bp.tryReserveUploadBytes,
	}
	row1 := queuedRow(opts, imageContent(make([]byte, 64)...))
	row1[eventIDField] = "ev-requeue"
	if got := bp.pendingUploadBytes.Load(); got != 64 {
		t.Fatalf("pendingUploadBytes after row1 reservation = %d, want 64", got)
	}

	// Depleted caller deadline expires during resolveUploads, deferring row1's
	// 64-byte upload onto row1[deferredUploadsKey].
	expiredCtx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	bp.resolveUploads(expiredCtx, false, []map[string]any{row1})
	cancel()

	// While row1's upload is deferred (before requeueDeferredUploads or
	// clearDeferredUploads), its 64-byte reservation must remain held so a
	// concurrent caller cannot reserve another 64 bytes and exceed the 64-byte cap.
	if got := bp.pendingUploadBytes.Load(); got != 64 {
		t.Fatalf("pendingUploadBytes while upload is deferred = %d, want 64", got)
	}
	if bp.tryReserveUploadBytes(1) {
		t.Fatal("tryReserveUploadBytes(1) succeeded while 64/64 bytes were held by deferred upload")
	}

	// Nothing has been requeued or dropped yet, so the failure is not logged.
	if logger.contains("GCS upload failed for event ev-requeue") {
		t.Fatalf("failure logged before requeueDeferredUploads:\n%s", logger)
	}

	// Requeuing the deferred upload for the next flush must keep the existing
	// 64-byte reservation without double-counting it to 128 bytes.
	bp.requeueDeferredUploads([]map[string]any{row1})
	retryLines, plainLines := 0, 0
	for line := range strings.SplitSeq(logger.String(), "\n") {
		if !strings.Contains(line, "event ev-requeue") {
			continue
		}
		if strings.Contains(line, "pending retry on next flush") {
			retryLines++
		} else if strings.Contains(line, "GCS upload failed") {
			plainLines++
		}
	}
	if retryLines != 1 || plainLines != 0 {
		t.Fatalf("requeueDeferredUploads retry lines = %d, plain lines = %d, want 1 and 0:\n%s", retryLines, plainLines, logger)
	}
	if got := bp.pendingUploadBytes.Load(); got != 64 {
		t.Fatalf("pendingUploadBytes after requeueDeferredUploads = %d, want 64 (must not exceed cap 64)", got)
	}

	// When the requeued batch is resolved on the next flush, the 64-byte
	// reservation is released back to 0.
	stalled.Store(false)
	bp.resolveUploads(context.Background(), false, []map[string]any{row1})
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Fatalf("pendingUploadBytes after retry resolveUploads = %d, want 0", got)
	}

	// Verify clearDeferredUploads also releases deferred bytes when a batch's
	// BigQuery write succeeds and the deferred uploads are discarded.
	row2 := queuedRow(opts, imageContent(make([]byte, 64)...))
	row2[eventIDField] = "ev-discard"
	if got := bp.pendingUploadBytes.Load(); got != 64 {
		t.Fatalf("pendingUploadBytes after row2 reservation = %d, want 64", got)
	}
	stalled.Store(true)
	expiredCtx2, cancel2 := context.WithTimeout(context.Background(), 25*time.Millisecond)
	bp.resolveUploads(expiredCtx2, false, []map[string]any{row2})
	cancel2()
	if got := bp.pendingUploadBytes.Load(); got != 64 {
		t.Fatalf("pendingUploadBytes while row2 upload is deferred = %d, want 64", got)
	}
	if logger.contains("GCS upload failed for event ev-discard") {
		t.Fatalf("discarded upload logged before clearDeferredUploads:\n%s", logger)
	}
	bp.clearDeferredUploads([]map[string]any{row2})
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Fatalf("pendingUploadBytes after clearDeferredUploads = %d, want 0", got)
	}
	discardPlain, discardRetry := 0, 0
	for line := range strings.SplitSeq(logger.String(), "\n") {
		if !strings.Contains(line, "event ev-discard") {
			continue
		}
		if strings.Contains(line, "pending retry") {
			discardRetry++
		} else if strings.Contains(line, "GCS upload failed") {
			discardPlain++
		}
	}
	if discardPlain != 1 || discardRetry != 0 {
		t.Errorf("clearDeferredUploads plain lines = %d, retry lines = %d, want 1 and 0:\n%s", discardPlain, discardRetry, logger)
	}
}

func TestResolveUploads_ConcurrencyLimitedBrownoutPausesUploads(t *testing.T) {
	var callNum atomic.Int32
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		n := callNum.Add(1)
		if n == 1 {
			select {
			case <-time.After(35 * time.Millisecond):
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		<-ctx.Done()
		return ctx.Err()
	}}
	bp := &BatchProcessor{
		config: Config{
			GCSOffloader:            off,
			GCSMaxConcurrentUploads: 1,
			GCSUploadTimeout:        60 * time.Millisecond,
			Logger:                  &recordingLogger{},
		},
	}
	opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 50}
	batch := []map[string]any{queuedRow(opts, imageContent(1), imageContent(2))}
	bp.resolveUploads(context.Background(), false, batch)
	if bp.uploadCooldownUntil.Load() == 0 {
		t.Fatalf("uploadCooldownUntil = 0 after concurrency-limited stall, want cooldown armed")
	}
}

func TestGCSOffloading_FailedMultiAttributeUploadTruncatedFlag(t *testing.T) {
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		return errors.New("gcs failure")
	}}
	bp := &BatchProcessor{
		config: Config{
			GCSOffloader:     off,
			GCSUploadTimeout: time.Minute,
			Logger:           &recordingLogger{},
		},
	}
	long := strings.Repeat("z", 200)
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{
				Text:       long,
				InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte{1, 2, 3}},
			},
		},
	}
	row := probeRow(t, content)
	bp.resolveUploads(context.Background(), false, []map[string]any{row})
	if row["is_truncated"] != true {
		t.Errorf("is_truncated = %v, want true after failed text upload with multi-attribute part", row["is_truncated"])
	}
}

func TestGCSOffloading_CloseLogsUploadFailuresUnderCallerDeadline(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	lis, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	requests := make(chan *storagepb.AppendRowsRequest, 16)
	storagepb.RegisterBigQueryWriteServer(srv, &delayedBigQueryWriteServer{
		delay:    10 * time.Millisecond,
		requests: requests,
	})
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	writeClient, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = writeClient.Close() })

	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		<-ctx.Done()
		return ctx.Err()
	}}
	logger := &recordingLogger{}
	cfg := DefaultConfig()
	cfg.BatchSize = 1
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 400 * time.Millisecond
	cfg.RetryConfig.MaxRetries = 0
	cfg.GCSOffloader = off
	cfg.Logger = logger

	bp, err := NewBatchProcessor(ctx, writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	bp.Start()

	row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3))
	row["event_type"] = EventLLMRequest
	row[eventIDField] = "ev-close-fail"
	bp.queue <- row

	bp.Close()

	rows := collectRows(t, requests, func(rows []map[string]any) bool { return len(rows) >= 1 })
	if len(rows) != 1 {
		t.Fatalf("written rows = %d, want 1", len(rows))
	}
	if got := contentPartsOf(t, rows[0])[0]["text"]; got != uploadFailedText {
		t.Errorf("part text = %v, want %q", got, uploadFailedText)
	}
	closePlain, closeRetry := 0, 0
	for line := range strings.SplitSeq(logger.String(), "\n") {
		if !strings.Contains(line, "event ev-close-fail") {
			continue
		}
		if strings.Contains(line, "pending retry") {
			closeRetry++
		} else if strings.Contains(line, "GCS upload failed") {
			closePlain++
		}
	}
	if closePlain != 1 || closeRetry != 0 {
		t.Errorf("Close plain lines = %d, retry lines = %d, want 1 and 0:\n%s", closePlain, closeRetry, logger)
	}
}

// TestGCSOffloading_CloseEarlyDropLoggerPanicReleasesUploadBytes checks that
// Close's early-drop branch releases the pending upload reservations before
// logging the warning, so that a Logger that panics on the warning cannot leak
// them. It queues three 16-byte pending uploads in a single batch that hits
// the rotation-cooldown early-drop branch, uses an always-panicking Logger,
// calls Close, recovers the panic, and asserts that pendingUploadBytes is 0.
func TestGCSOffloading_CloseEarlyDropLoggerPanicReleasesUploadBytes(t *testing.T) {
	ctx := context.Background()
	off := &fakeOffloader{}
	quietLogger := &recordingLogger{}
	cfg := exactlyOnceTestConfig()
	cfg.BatchFlushIntv = time.Hour
	cfg.BatchSize = 100
	cfg.GCSOffloader = off
	cfg.Logger = quietLogger
	cfg.ShutdownTimeout = 5 * time.Second
	srv := &scriptedCommittedStreamServer{
		onAppend: func(int, *storagepb.AppendRowsRequest) (*storagepb.AppendRowsResponse, error) {
			return nil, status.Error(codes.OutOfRange, "offset beyond end")
		},
	}
	bp, err := NewBatchProcessor(ctx, newScriptedWriteClient(t, srv), "", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor() error = %v", err)
	}

	// Establish rotation cooldown before Close so the 3 upload rows form
	// a batch that hits the cooldown early-drop branch immediately.
	if err := bp.writeBatch(ctx, []map[string]any{{"event_type": "B1"}}); err == nil {
		t.Fatal("writeBatch(B1) error = nil, want an offset error")
	}
	if !bp.streamWriter.isRotationInCooldown() {
		t.Fatal("rotation is not paused after the freshly rotated stream rejected the batch")
	}

	// Switch to the panicking Logger for Close.
	bp.config.Logger = alwaysPanicLogger{}

	batch := uploadRows(bp, 3)
	for _, row := range batch {
		bp.queue <- row
	}
	if got := bp.pendingUploadBytes.Load(); got != 48 {
		t.Fatalf("pendingUploadBytes before Close = %d, want 48", got)
	}

	// Close's early-drop branch logs a warning that the panicking Logger
	// turns into a panic; the panic is recovered, and the reservations
	// must already have been released.
	r := panicValue(func() { bp.Close() })
	if r == nil {
		t.Fatal("Close did not panic on the logger's warning")
	}
	if msg := fmt.Sprint(r); !strings.Contains(msg, "committed write stream rotation in cooldown") {
		t.Errorf("panic = %q, want it to contain the cooldown warning", msg)
	}
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes after Close panicked on the early-drop warning = %d, want 0", got)
	}
}

func TestGCSOffloading_CloseLogsFinalFailureOfBackgroundFlushUploadItCancels(t *testing.T) {
	requests := make(chan *storagepb.AppendRowsRequest, 8)
	wc := newScriptedWriteClient(t, &fakeBigQueryWriteServer{requests: requests})

	started := make(chan struct{})
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		select {
		case <-started:
		default:
			close(started)
		}
		<-ctx.Done()
		return ctx.Err()
	}}
	logger := &recordingLogger{}
	cfg := DefaultConfig()
	cfg.BatchSize = 1
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 2 * time.Second
	cfg.GCSUploadTimeout = 5 * time.Second
	cfg.GCSOffloader = off
	cfg.Logger = logger
	bp, err := NewBatchProcessor(context.Background(), wc, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	bp.Start()

	row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3))
	row["event_type"] = EventLLMRequest
	row[eventIDField] = "ev-bg"
	bp.Append(row) // reaches BatchSize, so a background flush starts uploading
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("upload did not start")
	}
	bp.Close() // cancels the in-flight upload of the background flush

	rows := collectRows(t, requests, func(rows []map[string]any) bool { return len(rows) >= 1 })
	if got := contentPartsOf(t, rows[0])[0]["text"]; got != uploadFailedText {
		t.Errorf("part text = %v, want %q", got, uploadFailedText)
	}
	var lines []string
	for _, l := range uploadFailureLines(logger) {
		if strings.Contains(l, "ev-bg") {
			lines = append(lines, l)
		}
	}
	want := "agentanalytics: GCS upload failed for event ev-bg part 0 (3 bytes): context canceled; keeping the part inline"
	if len(lines) != 1 || lines[0] != want {
		t.Errorf("upload failure lines = %q, want [%q]", lines, want)
	}
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes = %d, want 0", got)
	}
}

func TestGCSOffloading_LaterRunEndBatchLogsUploadFailures(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	logger := &recordingLogger{}
	// loggedDuringResolve records whether the failure of row 2's upload was
	// already logged when row 2's write reached BigQuery, which is after
	// resolveUploadsForBatch returned and before the write returns and
	// clearDeferredUploads settles the deferred upload.
	var loggedDuringResolve atomic.Bool
	requests := make(chan *storagepb.AppendRowsRequest, 16)
	writeClient := newScriptedWriteClient(t, &scriptedWriteServer{
		requests: requests,
		script: func(n int32) (time.Duration, error) {
			switch n {
			case 1:
				return 0, nil
			case 2:
				// Row 1's 1.6s write leaves at most 400ms, less than a
				// quarter (500ms) of the 2s ShutdownTimeout, so row 2's
				// run-end deadline does not pause the uploads
				// (pauseOnCallerDeadline is false) and its stalled upload is
				// deferred by resolveUploadsForBatch, then logged as a final
				// failure by clearDeferredUploads once row 2's write
				// succeeds.
				return 1600 * time.Millisecond, nil
			case 3:
				loggedDuringResolve.Store(logger.contains("ev-hang"))
				return 50 * time.Millisecond, nil
			default:
				return 50 * time.Millisecond, nil
			}
		},
	})

	var uploadCount atomic.Int32
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		if uploadCount.Add(1) == 1 {
			return nil
		}
		<-ctx.Done()
		return ctx.Err()
	}}
	cfg := DefaultConfig()
	cfg.BatchSize = 1
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 2 * time.Second
	cfg.RetryConfig.MaxRetries = 0
	cfg.GCSOffloader = off
	cfg.Logger = logger

	bp, err := NewBatchProcessor(ctx, writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	t.Cleanup(bp.Close)

	// Warm up the write stream before starting the run-end budget so gRPC
	// connection and stream setup do not consume ShutdownTimeout.
	warmCtx, cancelWarm := context.WithTimeout(ctx, 5*time.Second)
	defer cancelWarm()
	if err := bp.writeBatch(warmCtx, []map[string]any{{"event_type": EventUserMessageReceived}}); err != nil {
		t.Fatalf("warm-up writeBatch: %v", err)
	}
	<-requests

	bp.Start()

	row1 := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(1, 2, 3))
	row1["event_type"] = EventLLMRequest
	row1[eventIDField] = "ev-success"
	bp.queue <- row1

	row2 := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(4, 5, 6))
	row2["event_type"] = EventLLMRequest
	row2[eventIDField] = "ev-hang"
	bp.queue <- row2

	bp.flushRunEnd()

	rows := collectRows(t, requests, func(rows []map[string]any) bool { return len(rows) >= 2 })
	if len(rows) != 2 {
		t.Fatalf("written rows = %d, want 2", len(rows))
	}
	if got := contentPartsOf(t, rows[0])[0]["storage_mode"]; got != "GCS_REFERENCE" {
		t.Errorf("row 1 storage_mode = %v, want GCS_REFERENCE", got)
	}
	if got := contentPartsOf(t, rows[1])[0]["text"]; got != uploadFailedText {
		t.Errorf("row 2 text = %v, want %q", got, uploadFailedText)
	}
	if !logger.contains("GCS upload failed for event ev-hang") {
		t.Errorf("expected GCS upload failure logged for second batch, got:\n%s", logger)
	}
	if loggedDuringResolve.Load() {
		t.Error("the failure of row 2's upload was logged before row 2's write, want it deferred")
	}
	if got := bp.uploadCooldownUntil.Load(); got != 0 {
		t.Errorf("uploadCooldownUntil = %d, want 0: row 2's run-end deadline must not pause the uploads", got)
	}
	if got := len(bp.deferredBatch); got != 0 {
		t.Errorf("deferredBatch length = %d, want 0: row 2 was written", got)
	}
	// The second batch was written, so its upload is not retried.
	if logger.contains("pending retry on next flush") {
		t.Errorf("flushRunEnd logged a retry of an upload of a written batch, got:\n%s", logger)
	}
}

type reproPanicLogger struct{}

func (reproPanicLogger) Printf(format string, _ ...any) {
	if strings.Contains(format, "GCS upload failed") {
		panic("logger boom")
	}
}
func (reproPanicLogger) Println(...any) {}

func TestResolveUploads_PanickingLoggerReleasesDeferredReservation(t *testing.T) {
	newProcessor := func(off GCSContentOffloader, capBytes int64) *BatchProcessor {
		return &BatchProcessor{
			config: applyDefaults(Config{
				GCSOffloader:             off,
				GCSUploadTimeout:         time.Minute,
				GCSMaxConcurrentUploads:  2,
				GCSMaxPendingUploadBytes: capBytes,
				Logger:                   reproPanicLogger{},
			}),
			queue:                 make(chan map[string]any, 4),
			maxPendingUploadBytes: capBytes,
		}
	}

	t.Run("an in-flight log panic after deferring an earlier part", func(t *testing.T) {
		// The first part waits out the caller deadline and is deferred. The second
		// returns a non-gs URI, which is logged from applyUpload. That log panics
		// inside resolveUploads, after the deferred reservation is recorded.
		secondStarted := make(chan struct{})
		off := &fakeOffloader{
			hook: func(ctx context.Context, req GCSUploadRequest) error {
				if strings.Contains(req.Path, "_c1_p0") {
					close(secondStarted)
					return nil
				}
				<-ctx.Done()
				return ctx.Err()
			},
			uri: func(req GCSUploadRequest) string {
				if strings.Contains(req.Path, "_c1_p0") {
					return "https://example.com/not-gs"
				}
				return "gs://fake-bucket/" + req.Path
			},
		}
		bp := newProcessor(off, 128)
		opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}
		row := queuedRow(opts, imageContent(bytes.Repeat([]byte{1}, 64)...), imageContent(bytes.Repeat([]byte{2}, 64)...))
		if got := bp.pendingUploadBytes.Load(); got != 128 {
			t.Fatalf("pendingUploadBytes after reservation = %d, want 128", got)
		}
		if r := panicValue(func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			go func() {
				select {
				case <-secondStarted:
				case <-time.After(2 * time.Second):
				}
				cancel()
			}()
			bp.resolveUploads(ctx, false, []map[string]any{row})
		}); r == nil {
			t.Fatal("resolveUploads did not panic")
		}
		// The panic leaves flushRunEnd before the batch is requeued, so the deferred
		// row is gone too. Unwind must release those bytes as well.
		if got := bp.pendingUploadBytes.Load(); got != 0 {
			t.Fatalf("pendingUploadBytes = %d, want 0", got)
		}
		if _, ok := row[deferredUploadsKey]; ok {
			t.Fatal("row still holds deferredUploadsKey after resolveUploads panicked")
		}
	})

	t.Run("a deferred upload that is dropped", func(t *testing.T) {
		var stalled atomic.Bool
		stalled.Store(true)
		bp := newProcessor(stallingOffloader(&stalled), 64)
		opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}
		row := queuedRow(opts, imageContent(make([]byte, 64)...))
		// The deferred upload keeps its reservation, and its failure is not
		// logged, until it is dropped or requeued.
		deferUploads(t, bp, []map[string]any{row})
		if got := bp.pendingUploadBytes.Load(); got != 64 {
			t.Fatalf("pendingUploadBytes while the upload is deferred = %d, want 64", got)
		}
		if r := panicValue(func() { bp.clearDeferredUploads([]map[string]any{row}) }); r == nil {
			t.Fatal("clearDeferredUploads did not log the failure of the upload that it dropped")
		}
		if got := bp.pendingUploadBytes.Load(); got != 0 {
			t.Errorf("pendingUploadBytes = %d, want 0", got)
		}
	})

	t.Run("a failed upload that is logged at once", func(t *testing.T) {
		bp := newProcessor(&fakeOffloader{hook: func(context.Context, GCSUploadRequest) error {
			return errors.New("upload failed")
		}}, 64)
		opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}
		row := queuedRow(opts, imageContent(make([]byte, 64)...))
		if r := panicValue(func() { bp.resolveUploads(context.Background(), false, []map[string]any{row}) }); r == nil {
			t.Fatal("resolveUploads did not log the failed upload")
		}
		if got := bp.pendingUploadBytes.Load(); got != 0 {
			t.Errorf("pendingUploadBytes = %d, want 0", got)
		}
	})
}

func TestRequeueDeferredUploads_PanickingLoggerKeepsBatch(t *testing.T) {
	var stalled atomic.Bool
	stalled.Store(true)
	bp := &BatchProcessor{
		config: applyDefaults(Config{
			GCSOffloader:             stallingOffloader(&stalled),
			GCSUploadTimeout:         time.Minute,
			GCSMaxPendingUploadBytes: 64,
			Logger:                   reproPanicLogger{},
		}),
		queue:                 make(chan map[string]any, 4),
		maxPendingUploadBytes: 64,
	}
	opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}
	row := queuedRow(opts, imageContent(make([]byte, 64)...))
	row[eventIDField] = "ev-requeue-panic"
	expiredCtx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	bp.resolveUploads(expiredCtx, false, []map[string]any{row})
	cancel()
	if got := bp.pendingUploadBytes.Load(); got != 64 {
		t.Fatalf("pendingUploadBytes while upload is deferred = %d, want 64", got)
	}
	if r := panicValue(func() {
		bp.writeRunEndBatchLocked(expiredCtx, false, []map[string]any{row})
	}); r == nil {
		t.Fatal("writeRunEndBatchLocked did not panic on requeue log")
	}
	if len(bp.deferredBatch) != 1 {
		t.Fatalf("deferredBatch len = %d, want 1", len(bp.deferredBatch))
	}
	if got := bp.pendingUploadBytes.Load(); got != 64 {
		t.Fatalf("pendingUploadBytes after logger panic = %d, want 64", got)
	}
	saved := bp.deferredBatch
	bp.deferredBatch = nil
	stalled.Store(false)
	bp.config.Logger = &recordingLogger{}
	bp.resolveUploads(context.Background(), false, saved)
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Fatalf("pendingUploadBytes after resolving the saved batch = %d, want 0", got)
	}
}

func TestReleaseRowUploads_PanickingLoggerReleasesLaterRows(t *testing.T) {
	// resolveUploads consumed rawPartsKey (releasing the queued payload
	// reservations) but the deferred upload (deferredUploadsKey) still holds
	// its 64-byte reservation per row. releaseRowUploads drops a row via
	// releaseBatchUploads, which releases the deferred reservation before
	// logging. A panic on the first row's line must not keep the second row's
	// bytes when it is released separately.
	var stalled atomic.Bool
	stalled.Store(true)
	bp := &BatchProcessor{
		config: applyDefaults(Config{
			GCSOffloader:             stallingOffloader(&stalled),
			GCSUploadTimeout:         time.Minute,
			GCSMaxPendingUploadBytes: 128,
			Logger:                   reproPanicLogger{},
		}),
		queue:                 make(chan map[string]any, 4),
		maxPendingUploadBytes: 128,
	}
	opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}
	row1 := queuedRow(opts, imageContent(make([]byte, 64)...))
	row1[eventIDField] = "ev-write-fail-1"
	row2 := queuedRow(opts, imageContent(make([]byte, 64)...))
	row2[eventIDField] = "ev-write-fail-2"
	expiredCtx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	bp.resolveUploads(expiredCtx, false, []map[string]any{row1, row2})
	cancel()
	if got := bp.pendingUploadBytes.Load(); got != 128 {
		t.Fatalf("pendingUploadBytes while uploads are deferred = %d, want 128", got)
	}
	if r := panicValue(func() {
		bp.releaseRowUploads(row1)
		bp.config.logf("ERROR: Failed to write batch to BigQuery: %v", errors.New("write failed"))
	}); r == nil {
		t.Fatal("logger did not panic")
	}
	if got := bp.pendingUploadBytes.Load(); got != 64 {
		t.Fatalf("pendingUploadBytes after releasing row1 = %d, want 64", got)
	}
	if r := panicValue(func() {
		bp.releaseRowUploads(row2)
	}); r == nil {
		t.Fatal("logger did not panic on row2")
	}
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Fatalf("pendingUploadBytes after releasing row2 = %d, want 0", got)
	}
}

type callbackRaceLogger struct {
	n atomic.Int32
}

func (l *callbackRaceLogger) Printf(string, ...any) { l.n.Add(1) }
func (l *callbackRaceLogger) Println(...any)        { l.n.Add(1) }

func TestReleaseRowUploads_ConcurrentCallbackLogsDoNotRace(t *testing.T) {
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		<-ctx.Done()
		return ctx.Err()
	}}
	logger := &callbackRaceLogger{}
	bp := &BatchProcessor{
		config: applyDefaults(Config{
			GCSOffloader:             off,
			GCSUploadTimeout:         time.Minute,
			GCSMaxPendingUploadBytes: 64,
			BatchSize:                1000,
			Logger:                   logger,
		}),
		queue:                 make(chan map[string]any),
		maxPendingUploadBytes: 64,
	}
	opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}
	row := queuedRow(opts, imageContent(make([]byte, 64)...))
	expiredCtx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	bp.resolveUploads(expiredCtx, false, []map[string]any{row})
	cancel()
	if got := bp.pendingUploadBytes.Load(); got != 64 {
		t.Fatalf("pendingUploadBytes while upload is deferred = %d, want 64", got)
	}

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				bp.uploadBudgetWarned.Store(false)
				bp.tryReserveUploadBytes(1)
				bp.Append(map[string]any{eventIDField: "drop"})
			}
		}
	}()
	for range 8 {
		runtime.Gosched()
	}
	bp.releaseRowUploads(row)
	close(stop)
	wg.Wait()
	bp.clearDeferredUploads([]map[string]any{row})
}

func TestReleaseRowUploads_SharedLoggerKeepsDropLinesWithTheirProcessor(t *testing.T) {
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		<-ctx.Done()
		return ctx.Err()
	}}
	logger := &recordingLogger{}
	newProcessor := func() *BatchProcessor {
		return &BatchProcessor{
			config: applyDefaults(Config{
				GCSOffloader:             off,
				GCSUploadTimeout:         time.Minute,
				GCSMaxPendingUploadBytes: 64,
				Logger:                   logger,
			}),
			queue:                 make(chan map[string]any, 4),
			maxPendingUploadBytes: 64,
		}
	}
	other, bp := newProcessor(), newProcessor()
	opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}
	row := queuedRow(opts, imageContent(make([]byte, 64)...))
	row[eventIDField] = "ev-shared-logger"
	expiredCtx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	bp.resolveUploads(expiredCtx, false, []map[string]any{row})
	cancel()

	other.config.logf("ERROR: Failed to write batch to BigQuery: %v", errors.New("other processor"))
	bp.releaseRowUploads(row)
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Fatalf("pendingUploadBytes after releaseRowUploads = %d, want 0", got)
	}
	bp.clearDeferredUploads([]map[string]any{row})
	const want = "agentanalytics: GCS upload failed for event ev-shared-logger part 0 (64 bytes): context deadline exceeded; keeping the part inline"
	if diff := cmp.Diff([]string{want}, uploadFailureLines(logger)); diff != "" {
		t.Fatalf("upload failure lines mismatch (-want +got):\n%s", diff)
	}
}

func TestResolveUploads_CustomErrorUnderCallerDeadlineStaysDeferred(t *testing.T) {
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		<-ctx.Done()
		return errors.New("custom offloader failure")
	}}
	bp := &BatchProcessor{
		config: applyDefaults(Config{
			GCSOffloader:             off,
			GCSUploadTimeout:         time.Minute,
			GCSMaxPendingUploadBytes: 64,
			Logger:                   &recordingLogger{},
		}),
		queue:                 make(chan map[string]any, 4),
		maxPendingUploadBytes: 64,
	}
	opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}
	row := queuedRow(opts, imageContent(make([]byte, 64)...))
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	bp.resolveUploads(ctx, false, []map[string]any{row})
	if got := bp.pendingUploadBytes.Load(); got != 64 {
		t.Fatalf("pendingUploadBytes = %d, want 64", got)
	}
	if _, ok := row[rawPartsKey]; ok {
		t.Fatal("rawPartsKey set before requeueDeferredUploads")
	}
	bp.requeueDeferredUploads([]map[string]any{row})
	pending, ok := row[rawPartsKey].([]pendingUpload)
	if !ok || len(pending) != 1 {
		t.Fatalf("requeued uploads = %v, want the deferred part", row[rawPartsKey])
	}
}

// panicValue calls f and returns the value that it panics with, or nil.
func panicValue(f func()) (r any) {
	defer func() { r = recover() }()
	f()
	return nil
}

// stallingOffloader returns an offloader whose uploads wait until their
// context is done while stalled is set, and succeed otherwise.
func stallingOffloader(stalled *atomic.Bool) *fakeOffloader {
	return &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		if stalled.Load() {
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
}

// uploadRows returns n rows, ev-0 to ev-<n-1>, that each queue a 16-byte
// image for upload and reserve its bytes in bp.
func uploadRows(bp *BatchProcessor, n int) []map[string]any {
	opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}
	batch := make([]map[string]any, n)
	for i := range batch {
		batch[i] = queuedRow(opts, imageContent(make([]byte, 16)...))
		batch[i][eventIDField] = fmt.Sprintf("ev-%d", i)
	}
	return batch
}

// deferUploads resolves batch under a caller deadline that runs out while the
// uploads stall, as a run-end flush with a depleted budget does, and checks
// that every row has its upload deferred and that nothing about the deferred
// uploads was logged.
func deferUploads(t *testing.T, bp *BatchProcessor, batch []map[string]any) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	if r := panicValue(func() { bp.resolveUploads(ctx, false, batch) }); r != nil {
		t.Fatalf("resolveUploads logged the failure of a deferred upload: the logger panicked with %v", r)
	}
	for i, row := range batch {
		if d, _ := row[deferredUploadsKey].(*deferredRowUploads); d == nil || len(d.pending) == 0 {
			t.Fatalf("row %d has no deferred upload", i)
		}
	}
}

// uploadFailureLines returns the lines of l that report a failed upload.
func uploadFailureLines(l *recordingLogger) []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	var out []string
	for _, line := range l.lines {
		if strings.Contains(line, "GCS upload failed") {
			out = append(out, line)
		}
	}
	return out
}

// TestResolveUploads_DeferredUploadLogsRetryOnlyWhenRequeued checks that the
// failure of an upload that a depleted caller deadline cut short is logged
// once it is known whether the upload is retried, and only then: as pending a
// retry if the upload is requeued for the next flush, and as final if it is
// dropped.
func TestResolveUploads_DeferredUploadLogsRetryOnlyWhenRequeued(t *testing.T) {
	const failure = "agentanalytics: GCS upload failed for event ev-deferred part 0 (3 bytes): context deadline exceeded; keeping the part inline"
	tests := []struct {
		name    string
		settle  func(bp *BatchProcessor, batch []map[string]any)
		wantLog string
	}{
		{
			name:    "the upload is requeued",
			settle:  (*BatchProcessor).requeueDeferredUploads,
			wantLog: failure + " pending retry on next flush",
		},
		{
			name:    "the upload is dropped",
			settle:  (*BatchProcessor).clearDeferredUploads,
			wantLog: failure,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var stalled atomic.Bool
			stalled.Store(true)
			logger := &recordingLogger{}
			bp := &BatchProcessor{config: Config{GCSOffloader: stallingOffloader(&stalled), GCSUploadTimeout: time.Minute, Logger: logger}}
			row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10}, imageContent(1, 2, 3))
			row[eventIDField] = "ev-deferred"
			batch := []map[string]any{row}

			deferUploads(t, bp, batch)
			if got := uploadFailureLines(logger); len(got) != 0 {
				t.Errorf("resolveUploads logged %q before it was known whether the deferred upload is retried", got)
			}
			tt.settle(bp, batch)
			if diff := cmp.Diff([]string{tt.wantLog}, uploadFailureLines(logger)); diff != "" {
				t.Errorf("upload failure log mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestResolveUploads_CustomOffloaderErrorAfterCallerDeadlineIsDeferred checks
// that an offloader error that does not wrap context.DeadlineExceeded, returned
// once a depleted caller deadline has ended the uploads, defers the upload as a
// deadline error does, so that a requeued batch retries it.
func TestResolveUploads_CustomOffloaderErrorAfterCallerDeadlineIsDeferred(t *testing.T) {
	var stalled atomic.Bool
	stalled.Store(true)
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		if stalled.Load() {
			<-ctx.Done()
			return errors.New("custom timeout error")
		}
		return nil
	}}
	logger := &recordingLogger{}
	bp := &BatchProcessor{config: Config{GCSOffloader: off, GCSUploadTimeout: time.Minute, Logger: logger}}
	row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10}, imageContent(1, 2, 3))
	row[eventIDField] = "ev-custom"
	batch := []map[string]any{row}

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	bp.resolveUploads(ctx, false, batch)
	cancel()
	if d, _ := row[deferredUploadsKey].(*deferredRowUploads); d == nil || len(d.pending) != 1 {
		t.Fatalf("the upload that failed with a custom error after the caller deadline ran out was not deferred; log:\n%s", logger)
	}

	bp.requeueDeferredUploads(batch)
	if pending, _ := row[rawPartsKey].([]pendingUpload); len(pending) != 1 {
		t.Fatalf("the row has %d uploads queued after requeueDeferredUploads, want 1", len(pending))
	}
	want := "agentanalytics: GCS upload failed for event ev-custom part 0 (3 bytes): error type *errors.errorString; keeping the part inline pending retry on next flush"
	if diff := cmp.Diff([]string{want}, uploadFailureLines(logger)); diff != "" {
		t.Errorf("upload failure log mismatch (-want +got):\n%s", diff)
	}

	stalled.Store(false)
	bp.resolveUploads(context.Background(), false, batch)
	if got := contentPartsOf(t, row)[0]["storage_mode"]; got != "GCS_REFERENCE" {
		t.Errorf("storage_mode after the retry = %v, want GCS_REFERENCE", got)
	}
	if got := len(off.uploads()); got != 2 {
		t.Errorf("the offloader was called %d times, want 2: the cut-short upload and its retry", got)
	}
}

// TestClearDeferredUploads_PanickingLoggerReleasesEveryRow checks that the
// reservations of the deferred uploads of every row of a batch are released
// before any of their failures is logged, so that a Logger that panics on the
// first of them leaks none.
func TestClearDeferredUploads_PanickingLoggerReleasesEveryRow(t *testing.T) {
	var stalled atomic.Bool
	stalled.Store(true)
	bp := &BatchProcessor{config: Config{GCSOffloader: stallingOffloader(&stalled), GCSUploadTimeout: time.Minute, Logger: reproPanicLogger{}}}
	batch := uploadRows(bp, 3)
	deferUploads(t, bp, batch)
	if got := bp.pendingUploadBytes.Load(); got != 48 {
		t.Fatalf("pendingUploadBytes while the uploads are deferred = %d, want 48", got)
	}

	if r := panicValue(func() { bp.clearDeferredUploads(batch) }); r == nil {
		t.Fatal("clearDeferredUploads did not log the failures of the uploads that it dropped")
	}
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes after the logger panicked = %d, want 0", got)
	}
	for i, row := range batch {
		if _, ok := row[deferredUploadsKey]; ok {
			t.Errorf("row %d still holds its deferred uploads", i)
		}
	}
}

// TestRequeueDeferredUploads_PanickingLoggerRequeuesEveryRow checks that every
// row of a batch is requeued, keeping the reservations of its uploads, before
// the failures of the deferred uploads are logged, so that a Logger that
// panics on the first of them leaves no row without its uploads.
func TestRequeueDeferredUploads_PanickingLoggerRequeuesEveryRow(t *testing.T) {
	var stalled atomic.Bool
	stalled.Store(true)
	bp := &BatchProcessor{config: Config{GCSOffloader: stallingOffloader(&stalled), GCSUploadTimeout: time.Minute, Logger: reproPanicLogger{}}}
	batch := uploadRows(bp, 3)
	deferUploads(t, bp, batch)

	if r := panicValue(func() { bp.requeueDeferredUploads(batch) }); r == nil {
		t.Fatal("requeueDeferredUploads did not log the failures of the uploads that it requeued")
	}
	for i, row := range batch {
		if pending, _ := row[rawPartsKey].([]pendingUpload); len(pending) != 1 {
			t.Errorf("row %d has %d uploads queued after the logger panicked, want 1", i, len(pending))
		}
		if got := contentPartsOf(t, row)[0]["text"]; got != inlineDataText {
			t.Errorf("row %d part text = %v, want %q until the upload is retried", i, got, inlineDataText)
		}
	}
	if got := bp.pendingUploadBytes.Load(); got != 48 {
		t.Errorf("pendingUploadBytes after the logger panicked = %d, want 48, kept for the retry", got)
	}

	// The next flush retries every upload and releases the reservations.
	stalled.Store(false)
	bp.resolveUploads(context.Background(), false, batch)
	for i, row := range batch {
		if got := contentPartsOf(t, row)[0]["storage_mode"]; got != "GCS_REFERENCE" {
			t.Errorf("row %d storage_mode after the retry = %v, want GCS_REFERENCE", i, got)
		}
	}
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes after the retry = %d, want 0", got)
	}
}

// TestReleaseBatchUploads_PanickingLoggerReleasesEveryRow checks that dropping
// a batch releases the reservations of the payloads still queued on its rows,
// and of its deferred uploads, before the failures of the deferred uploads are
// logged, so that a Logger that panics on the first of them leaks nothing.
func TestReleaseBatchUploads_PanickingLoggerReleasesEveryRow(t *testing.T) {
	var stalled atomic.Bool
	stalled.Store(true)
	bp := &BatchProcessor{config: Config{GCSOffloader: stallingOffloader(&stalled), GCSUploadTimeout: time.Minute, Logger: reproPanicLogger{}}}
	batch := uploadRows(bp, 3)
	// The first two rows have deferred uploads; the last still has its
	// payload queued.
	deferUploads(t, bp, batch[:2])
	if got := bp.pendingUploadBytes.Load(); got != 48 {
		t.Fatalf("pendingUploadBytes before the batch is dropped = %d, want 48", got)
	}

	if r := panicValue(func() { bp.releaseBatchUploads(batch) }); r == nil {
		t.Fatal("releaseBatchUploads did not log the failures of the deferred uploads that it dropped")
	}
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes after the logger panicked = %d, want 0", got)
	}
	for i, row := range batch {
		for _, key := range []string{rawPartsKey, deferredUploadsKey} {
			if _, ok := row[key]; ok {
				t.Errorf("row %d still has %s", i, key)
			}
		}
	}
}

// TestGCSOffloading_FlushRunEndKeepsDeferredBatchWhenLoggerPanics checks that a
// run-end flush whose BigQuery write runs out of its budget keeps the batch in
// deferredBatch before it logs that the batch's deferred uploads are pending a
// retry, so that a Logger that panics on that line loses neither the batch nor
// the reservations of its uploads.
func TestGCSOffloading_FlushRunEndKeepsDeferredBatchWhenLoggerPanics(t *testing.T) {
	requests := make(chan *storagepb.AppendRowsRequest, 16)
	var serverDelay atomic.Int64
	writeClient := newScriptedWriteClient(t, &scriptedWriteServer{
		requests: requests,
		script: func(int32) (time.Duration, error) {
			return time.Duration(serverDelay.Load()), nil
		},
	})

	var stalled atomic.Bool
	stalled.Store(true)
	cfg := DefaultConfig()
	cfg.BatchSize = 1
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = time.Second
	cfg.RetryConfig.MaxRetries = 0
	cfg.GCSOffloader = stallingOffloader(&stalled)
	cfg.Logger = reproPanicLogger{}
	bp, err := NewBatchProcessor(context.Background(), writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	t.Cleanup(func() {
		stalled.Store(false)
		serverDelay.Store(0)
		_ = panicValue(bp.Close)
	})

	// Warm up the write stream before starting the run-end budget so gRPC
	// connection and stream setup do not consume ShutdownTimeout.
	warmCtx, cancelWarm := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancelWarm()
	if err := bp.writeBatch(warmCtx, []map[string]any{{"event_type": EventInvocationStarting}}); err != nil {
		t.Fatalf("warm-up writeBatch: %v", err)
	}
	<-requests

	serverDelay.Store(int64(800 * time.Millisecond))

	// Batch 0 has no uploads, and its 800ms write leaves ~200ms, less than a
	// quarter (250ms) of the 1s ShutdownTimeout: batch 1 starts with
	// pauseOnCallerDeadline false, so its stalled upload is deferred without
	// pausing uploads, and its 800ms write runs out of the budget, keeping
	// batch 1 for the next flush.
	bp.queue <- map[string]any{"event_type": EventUserMessageReceived}
	row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}, imageContent(make([]byte, 64)...))
	row["event_type"] = EventLLMRequest
	bp.queue <- row

	if r := panicValue(bp.flushRunEnd); r == nil {
		t.Fatal("flushRunEnd did not log the failure of the requeued upload")
	}
	if got := bp.uploadCooldownUntil.Load(); got != 0 {
		t.Errorf("uploadCooldownUntil = %d, want 0: the deferred upload must not pause the uploads", got)
	}
	if len(bp.deferredBatch) != 1 {
		t.Fatalf("deferredBatch length after the logger panicked = %d, want 1", len(bp.deferredBatch))
	}
	if got := bp.deferredBatch[0]["event_type"]; got != EventLLMRequest {
		t.Fatalf("deferredBatch[0] event_type = %v, want %s", got, EventLLMRequest)
	}
	if pending, _ := row[rawPartsKey].([]pendingUpload); len(pending) != 1 {
		t.Errorf("the kept row has %d uploads queued, want its deferred upload", len(pending))
	}
	if got := bp.pendingUploadBytes.Load(); got != 64 {
		t.Errorf("pendingUploadBytes after the logger panicked = %d, want 64, kept for the retry", got)
	}

	// Once Cloud Storage and BigQuery recover, the next flush writes the kept
	// batch, retrying its upload.
	stalled.Store(false)
	serverDelay.Store(0)
	bp.flush()
	if len(bp.deferredBatch) != 0 {
		t.Fatalf("deferredBatch length after flush = %d, want 0", len(bp.deferredBatch))
	}
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes after flush = %d, want 0", got)
	}

	// Under at-least-once delivery, the timed-out INLINE attempt for batch 1
	// may also be recorded before the retried GCS_REFERENCE write.
	rows := collectRows(t, requests, func(rows []map[string]any) bool {
		return len(rows) >= 2
	})
	if contentPartsOf(t, rows[len(rows)-1])[0]["storage_mode"] == "INLINE" {
		select {
		case req := <-requests:
			rows = append(rows, decodeArrowRows(t, req)...)
		case <-time.After(250 * time.Millisecond):
		}
	}
	if len(rows) != 2 && len(rows) != 3 {
		t.Fatalf("persisted rows = %d, want 2 or 3", len(rows))
	}
	if got := rows[0]["event_type"]; got != EventUserMessageReceived {
		t.Errorf("rows[0] event_type = %v, want %s", got, EventUserMessageReceived)
	}
	if len(rows) == 3 {
		if got := rows[1]["event_type"]; got != EventLLMRequest {
			t.Errorf("rows[1] event_type = %v, want %s", got, EventLLMRequest)
		}
	}
	if got := rows[len(rows)-1]["event_type"]; got != EventLLMRequest {
		t.Errorf("last row event_type = %v, want %s", got, EventLLMRequest)
	}
	if got := contentPartsOf(t, rows[len(rows)-1])[0]["storage_mode"]; got != "GCS_REFERENCE" {
		t.Errorf("the retried part has storage_mode %v, want GCS_REFERENCE", got)
	}
}

// scriptedWriteServer answers each AppendRows request as script says for its
// number, counting from 1: after a delay, unless the stream ends first, or
// with an error that ends the stream. If respond is set, the answer is what
// respond returns for the number, if not nil. It records the requests that it
// answers.
type scriptedWriteServer struct {
	storagepb.UnimplementedBigQueryWriteServer
	script   func(n int32) (time.Duration, error)
	respond  func(n int32) *storagepb.AppendRowsResponse
	count    atomic.Int32
	requests chan *storagepb.AppendRowsRequest
}

func (s *scriptedWriteServer) AppendRows(stream storagepb.BigQueryWrite_AppendRowsServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		n := s.count.Add(1)
		delay, err := s.script(n)
		if err != nil {
			return err
		}
		select {
		case <-time.After(delay):
		case <-stream.Context().Done():
			return stream.Context().Err()
		}
		s.requests <- req
		resp := &storagepb.AppendRowsResponse{}
		if s.respond != nil {
			if r := s.respond(n); r != nil {
				resp = r
			}
		}
		if err := stream.Send(resp); err != nil {
			return err
		}
	}
}

// TestResolveUploads_CanceledCallerContextDefersUploads checks that an upload
// that a canceled caller context cuts short without pausing uploads across
// flushes is deferred, as one that a depleted deadline cuts short is, so that
// a requeued batch retries it.
func TestResolveUploads_CanceledCallerContextDefersUploads(t *testing.T) {
	started := make(chan struct{})
	var calls atomic.Int32
	off := &fakeOffloader{hook: func(ctx context.Context, _ GCSUploadRequest) error {
		if calls.Add(1) == 1 {
			close(started)
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
	logger := &recordingLogger{}
	bp := &BatchProcessor{config: Config{GCSOffloader: off, GCSUploadTimeout: time.Minute, Logger: logger}}
	row := queuedRow(partsOptions{maxLen: 1000, offload: true, inlineLimit: 10}, imageContent(1, 2, 3))
	row[eventIDField] = "ev-canceled"
	batch := []map[string]any{row}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-started
		cancel()
	}()
	bp.resolveUploads(ctx, false, batch)
	if d, _ := row[deferredUploadsKey].(*deferredRowUploads); d == nil || len(d.pending) != 1 {
		t.Fatalf("the upload that a canceled caller context cut short was not deferred; log:\n%s", logger)
	}

	bp.requeueDeferredUploads(batch)
	want := "agentanalytics: GCS upload failed for event ev-canceled part 0 (3 bytes): context canceled; keeping the part inline pending retry on next flush"
	if diff := cmp.Diff([]string{want}, uploadFailureLines(logger)); diff != "" {
		t.Errorf("upload failure log mismatch (-want +got):\n%s", diff)
	}
	bp.resolveUploads(context.Background(), false, batch)
	if got := contentPartsOf(t, row)[0]["storage_mode"]; got != "GCS_REFERENCE" {
		t.Errorf("storage_mode after the retry = %v, want GCS_REFERENCE", got)
	}
}

// TestGCSOffloading_CloseRetriesUploadThatItCancelsInRunEndFlush checks that
// an upload that Close cancels while a run-end flush is uploading is deferred:
// when the BigQuery write of its batch then runs out of the budget of the
// flush, the flush requeues the upload, logs it once as pending a retry, and
// keeps the batch, and Close retries the upload when it writes the batch.
func TestGCSOffloading_CloseRetriesUploadThatItCancelsInRunEndFlush(t *testing.T) {
	requests := make(chan *storagepb.AppendRowsRequest, 16)
	writeClient := newScriptedWriteClient(t, &scriptedWriteServer{
		requests: requests,
		script: func(n int32) (time.Duration, error) {
			switch n {
			case 1:
				// The write of row A leaves ~1.2s of the 3s budget (~600ms
				// for row B's upload window), leaving margin for Close to
				// cancel row B's in-flight upload before its upload deadline
				// expires; because Close's cancellation rather than the
				// upload deadline ends the upload, it does not pause uploads
				// and is deferred.
				return 1800 * time.Millisecond, nil
			case 2:
				// The write of the batch of row B outlasts the budget.
				return time.Minute, nil
			default:
				return 0, nil
			}
		},
	})
	bStarted := make(chan struct{})
	var bCalls atomic.Int32
	off := &fakeOffloader{hook: func(ctx context.Context, req GCSUploadRequest) error {
		if strings.Contains(req.Path, "_ev-b_") && bCalls.Add(1) == 1 {
			close(bStarted)
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
	logger := &recordingLogger{}
	cfg := DefaultConfig()
	cfg.BatchSize = 1
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 3 * time.Second
	cfg.RetryConfig.MaxRetries = 0
	cfg.GCSOffloader = off
	cfg.Logger = logger
	bp, err := NewBatchProcessor(context.Background(), writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	t.Cleanup(bp.Close)

	opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}
	for _, id := range []string{"ev-a", "ev-b"} {
		row := queuedRow(opts, imageContent(1, 2, 3))
		row["event_type"] = EventLLMRequest
		row[eventIDField] = id
		bp.queue <- row
	}

	// Close cancels the uploads of the run-end flush while the upload of row
	// B is in flight, and then waits for the flush to finish.
	closed := make(chan struct{})
	go func() {
		<-bStarted
		bp.Close()
		close(closed)
	}()
	bp.flushRunEnd()
	select {
	case <-closed:
	case <-time.After(10 * time.Second):
		t.Fatal("Close did not return")
	}

	rows := collectRows(t, requests, func(rows []map[string]any) bool { return len(rows) >= 2 })
	var rowB map[string]any
	for _, row := range rows {
		if row[eventIDField] == "ev-b" {
			rowB = row
		}
	}
	if rowB == nil {
		t.Fatalf("row B was not written; event ids: %v, %v", rows[0][eventIDField], rows[1][eventIDField])
	}
	if got := contentPartsOf(t, rowB)[0]["storage_mode"]; got != "GCS_REFERENCE" {
		t.Errorf("row B storage_mode = %v, want GCS_REFERENCE: Close retries the upload that it canceled", got)
	}
	if got := len(off.uploads()); got != 3 {
		t.Errorf("the offloader was called %d times, want 3: row A, row B, and the retry of row B", got)
	}
	want := "agentanalytics: GCS upload failed for event ev-b part 0 (3 bytes): context canceled; keeping the part inline pending retry on next flush"
	if diff := cmp.Diff([]string{want}, uploadFailureLines(logger)); diff != "" {
		t.Errorf("upload failure log mismatch (-want +got):\n%s", diff)
	}
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes after Close = %d, want 0", got)
	}
	if got := bp.uploadCooldownUntil.Load(); got != 0 {
		t.Errorf("uploadCooldownUntil after Close = %d, want 0: canceling an upload must not pause the uploads", got)
	}
}

// TestGCSOffloading_FlushRunEndDropPathReleasesAllUploadsWhenLoggerPanics
// checks that when BigQuery rejects the write of a run-end batch whose uploads
// were deferred, flushRunEnd releases the reservations of the deferred uploads
// of every row of the batch before it logs their failures, so that a Logger
// that panics on the first of them leaks none.
func TestGCSOffloading_FlushRunEndDropPathReleasesAllUploadsWhenLoggerPanics(t *testing.T) {
	requests := make(chan *storagepb.AppendRowsRequest, 16)
	writeClient := newScriptedWriteClient(t, &scriptedWriteServer{
		requests: requests,
		script: func(n int32) (time.Duration, error) {
			switch n {
			case 1:
				return 0, nil
			case 2:
				// Batch A's 1.6s write leaves less than a quarter (500ms) of
				// the 2s budget, so batch B's run-end deadline cannot pause
				// the uploads and its stalled uploads are deferred, while
				// leaving time for batch B's rejected write.
				return 1600 * time.Millisecond, nil
			default:
				return 0, status.Error(codes.InvalidArgument, "rejected")
			}
		},
	})
	var closing atomic.Bool
	off := &fakeOffloader{hook: func(ctx context.Context, req GCSUploadRequest) error {
		if !closing.Load() && strings.Contains(req.Path, "_ev-b") {
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
	cfg := DefaultConfig()
	cfg.BatchSize = 3
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 2 * time.Second
	cfg.RetryConfig.MaxRetries = 0
	cfg.GCSOffloader = off
	cfg.Logger = reproPanicLogger{}
	bp, err := NewBatchProcessor(context.Background(), writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	t.Cleanup(func() {
		closing.Store(true)
		_ = panicValue(bp.Close)
	})

	// Warm up the write stream before starting the run-end budget so gRPC
	// connection and stream setup do not consume ShutdownTimeout.
	warmCtx, cancelWarm := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancelWarm()
	if err := bp.writeBatch(warmCtx, []map[string]any{{"event_type": EventInvocationStarting}}); err != nil {
		t.Fatalf("warm-up writeBatch: %v", err)
	}
	<-requests

	// Batch A uploads its rows; batch B has its uploads deferred and its
	// write rejected, which drops it.
	opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}
	for _, id := range []string{"ev-a0", "ev-a1", "ev-a2", "ev-b0", "ev-b1", "ev-b2"} {
		row := queuedRow(opts, imageContent(make([]byte, 16)...))
		row["event_type"] = EventLLMRequest
		row[eventIDField] = id
		bp.queue <- row
	}

	if r := panicValue(bp.flushRunEnd); r == nil {
		t.Fatal("flushRunEnd did not log the failures of the deferred uploads of the rejected batch")
	}
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes after the logger panicked on the drop path = %d, want 0", got)
	}
	if got := len(bp.deferredBatch); got != 0 {
		t.Errorf("deferredBatch length = %d, want 0: a rejected batch is dropped", got)
	}
	if got := bp.DropStats().AppendFailureDrops; got != 3 {
		t.Errorf("AppendFailureDrops = %d, want 3, the rows of the rejected batch", got)
	}
}

// TestResolveUploads_UnsettledDeferredUploadsLoseNoReservationWhenLoggerPanics
// checks that resolveUploads drops the deferred uploads that a caller left
// unsettled before it takes over the reservations of the payloads queued on
// the batch, so that a Logger that panics on their failures leaves every
// reservation with a row that holds it.
func TestResolveUploads_UnsettledDeferredUploadsLoseNoReservationWhenLoggerPanics(t *testing.T) {
	var stalled atomic.Bool
	stalled.Store(true)
	bp := &BatchProcessor{config: Config{GCSOffloader: stallingOffloader(&stalled), GCSUploadTimeout: time.Minute, Logger: reproPanicLogger{}}}
	batch := uploadRows(bp, 2)
	// The upload of the second row is deferred and left unsettled, while the
	// first row still has its payload queued.
	deferUploads(t, bp, batch[1:])
	if got := bp.pendingUploadBytes.Load(); got != 32 {
		t.Fatalf("pendingUploadBytes before the batch is resolved = %d, want 32", got)
	}

	stalled.Store(false)
	if r := panicValue(func() { bp.resolveUploads(context.Background(), false, batch) }); r == nil {
		t.Fatal("resolveUploads did not log the failure of the unsettled deferred upload that it dropped")
	}
	// Right after the recovered panic, before the batch is released, the
	// first row's still-queued payload holds 16 bytes; the second row's
	// 16-byte stale deferred reservation was released by clearDeferredUploads
	// inside resolveUploads, so 16 bytes remain reserved, not 32.
	if got := bp.pendingUploadBytes.Load(); got != 16 {
		t.Errorf("pendingUploadBytes after the logger panicked, before releaseBatchUploads = %d, want 16", got)
	}
	// Dropping the batch releases what its rows still hold, which must be
	// every reservation that is left.
	bp.releaseBatchUploads(batch)
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes after the logger panicked and the batch was dropped = %d, want 0", got)
	}
}

// alwaysPanicLogger panics on every line that it is asked to log.
type alwaysPanicLogger struct{}

func (alwaysPanicLogger) Printf(format string, _ ...any) { panic("logger boom: " + format) }
func (alwaysPanicLogger) Println(...any)                 { panic("logger boom") }

// TestGCSOffloading_FlushRunEndReleasesDeferredUploadsWhenAppendLogPanics
// checks that when the BigQuery write of a run-end batch whose uploads were
// deferred panics, as it does when BigQuery answers with an in-band error and
// the Logger panics on the "BigQuery response" line, the reservations of the
// deferred uploads are released although they were never settled.
func TestGCSOffloading_FlushRunEndReleasesDeferredUploadsWhenAppendLogPanics(t *testing.T) {
	requests := make(chan *storagepb.AppendRowsRequest, 16)
	writeClient := newScriptedWriteClient(t, &scriptedWriteServer{
		requests: requests,
		script: func(n int32) (time.Duration, error) {
			if n == 2 {
				// Batch A's 1.6s write leaves less than a quarter (500ms) of
				// the 2s budget, so batch B's run-end deadline cannot pause
				// the uploads and its stalled uploads are deferred, while
				// leaving time for batch B's rejected write.
				return 1600 * time.Millisecond, nil
			}
			return 0, nil
		},
		respond: func(n int32) *storagepb.AppendRowsResponse {
			if n <= 2 {
				return nil
			}
			return &storagepb.AppendRowsResponse{
				Response: &storagepb.AppendRowsResponse_Error{
					Error: &statuspb.Status{Code: int32(codes.InvalidArgument), Message: "rejected"},
				},
			}
		},
	})
	var closing atomic.Bool
	off := &fakeOffloader{hook: func(ctx context.Context, req GCSUploadRequest) error {
		if !closing.Load() && strings.Contains(req.Path, "_ev-b") {
			<-ctx.Done()
			return ctx.Err()
		}
		return nil
	}}
	cfg := DefaultConfig()
	cfg.BatchSize = 3
	cfg.BatchFlushIntv = time.Hour
	cfg.ShutdownTimeout = 2 * time.Second
	cfg.RetryConfig.MaxRetries = 0
	cfg.GCSOffloader = off
	cfg.Logger = alwaysPanicLogger{}
	bp, err := NewBatchProcessor(context.Background(), writeClient, "projects/p/datasets/d/tables/t/_default", cfg)
	if err != nil {
		t.Fatalf("NewBatchProcessor: %v", err)
	}
	t.Cleanup(func() {
		closing.Store(true)
		_ = panicValue(bp.Close)
	})

	// Warm up the write stream before starting the run-end budget so gRPC
	// connection and stream setup do not consume ShutdownTimeout.
	warmCtx, cancelWarm := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancelWarm()
	if err := bp.writeBatch(warmCtx, []map[string]any{{"event_type": EventInvocationStarting}}); err != nil {
		t.Fatalf("warm-up writeBatch: %v", err)
	}
	<-requests

	// Batch A uploads its rows; batch B has its three 16-byte uploads
	// deferred, and its write is answered with an in-band error, whose log
	// line panics before the deferred uploads are settled.
	opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 10, reserveBytes: bp.tryReserveUploadBytes}
	for _, id := range []string{"ev-a0", "ev-a1", "ev-a2", "ev-b0", "ev-b1", "ev-b2"} {
		row := queuedRow(opts, imageContent(make([]byte, 16)...))
		row["event_type"] = EventLLMRequest
		row[eventIDField] = id
		bp.queue <- row
	}

	r := panicValue(bp.flushRunEnd)
	if s, _ := r.(string); !strings.Contains(s, "BigQuery response") {
		t.Fatalf("flushRunEnd panicked with %v, want the panic of the BigQuery response log line", r)
	}
	if got := bp.pendingUploadBytes.Load(); got != 0 {
		t.Errorf("pendingUploadBytes after the append panicked = %d, want 0", got)
	}
}

func TestGCSObjectPath_SanitizationAndBounds(t *testing.T) {
	// Pin Cloud Storage's external 1,024-byte object-name limit and 65-byte
	// "~<sha256>" suffix length independently of production constants so any
	// drift in maxGCSObjectNameBytes or pathSegmentHashSuffixLen fails tests.
	const (
		wantObjectLimit   = 1024
		wantHashSuffixLen = 65
	)
	ts := time.Date(2026, 3, 4, 5, 6, 7, 0, time.UTC)

	t.Run("non-ASCII and sanitized ids do not collide", func(t *testing.T) {
		pairs := [][2]string{
			{"é", "ü"},
			{"é", "__4a99557e"},
			{"é", "_" + pathSegmentHashSuffix("é")},
			{"session-日本", "session-韓国"},
			{"日本", "韓国"},
			{"日本", "__"},
			{".", ".."},
			{".", ""},
			{"..", ""},
			{"é?*", "///"},
			{"é?*", "___"},
			{"a/b", "a_b"},
			{"a/b", "a_b_c14cddc0"},
			{"a/b", "a_b" + pathSegmentHashSuffix("a/b")},
			{strings.Repeat("a", 128) + "5606", strings.Repeat("a", 128) + "91879"},
			{"\uac01\uac01\uac02\uac05\uac09", "\uac01\uac05\uac0f\uac08\uac07"},
		}
		for _, pair := range pairs {
			p0 := gcsObjectPath("logs", ts, "trace-1", "span-1", pair[0], 0, 0, ".txt")
			p1 := gcsObjectPath("logs", ts, "trace-1", "span-1", pair[1], 0, 0, ".txt")
			if p0 == p1 {
				t.Errorf("eventID %q and %q collided on GCS object path %q", pair[0], pair[1], p0)
			}
		}
	})

	t.Run("distinct non-ASCII event ids upload both payloads", func(t *testing.T) {
		ctx := context.Background()
		pairs := [][2]string{
			{"session-日本", "session-韓国"},
			{"é", "__4a99557e"},
			{"é", "_" + pathSegmentHashSuffix("é")},
			{"a/b", "a_b_c14cddc0"},
			{"a/b", "a_b" + pathSegmentHashSuffix("a/b")},
			{strings.Repeat("a", 128) + "5606", strings.Repeat("a", 128) + "91879"},
			{"\uac01\uac01\uac02\uac05\uac09", "\uac01\uac05\uac0f\uac08\uac07"},
		}
		for _, pair := range pairs {
			fs := newFakeStorage(t)
			offloader, err := newGCSOffloader(ctx, "test-bucket", fs.clientOptions()...)
			if err != nil {
				t.Fatalf("newGCSOffloader() error: %v", err)
			}
			t.Cleanup(func() { _ = offloader.Close() })

			bp := &BatchProcessor{config: Config{GCSOffloader: offloader, GCSPathPrefix: "logs", ConnectionID: "us.conn"}}
			opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 5}
			row0 := queuedRow(opts, &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "payload-first-123456"}}})
			row0[timestampField] = ts
			row0["trace_id"] = "trace-1"
			row0["span_id"] = "span-1"
			row0[eventIDField] = pair[0]

			row1 := queuedRow(opts, &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "payload-second-654321"}}})
			row1[timestampField] = ts
			row1["trace_id"] = "trace-1"
			row1["span_id"] = "span-1"
			row1[eventIDField] = pair[1]

			bp.resolveUploads(ctx, false, []map[string]any{row0, row1})

			uri0, _ := row0[columnContentParts].([]map[string]any)[0]["uri"].(string)
			uri1, _ := row1[columnContentParts].([]map[string]any)[0]["uri"].(string)
			if uri0 == "" || uri1 == "" || uri0 == uri1 {
				t.Fatalf("pair %q vs %q got URIs %q and %q, want two distinct non-empty gs:// URIs", pair[0], pair[1], uri0, uri1)
			}
			obj0, ok0 := fs.object("test-bucket", strings.TrimPrefix(uri0, "gs://test-bucket/"))
			obj1, ok1 := fs.object("test-bucket", strings.TrimPrefix(uri1, "gs://test-bucket/"))
			if !ok0 || !ok1 || string(obj0.data) != "payload-first-123456" || string(obj1.data) != "payload-second-654321" {
				t.Errorf("pair %q vs %q stored (%v:%q, %v:%q), want both payloads stored distinctly",
					pair[0], pair[1], ok0, obj0.data, ok1, obj1.data)
			}
		}
	})

	t.Run("long ids and prefixes stay within 1024 bytes without colliding", func(t *testing.T) {
		longA := strings.Repeat("a", 1100)
		longB := strings.Repeat("a", 1000) + "-b-" + strings.Repeat("z", 100)
		longMultiSegPrefix := strings.Repeat("segment123/", 120)
		longUTF8 := strings.Repeat("日本", 600)

		cases := []struct {
			name                     string
			prefix                   string
			traceID, spanID, eventID string
			ext                      string
		}{
			{"long_event_id", "logs", "trace", "span", longA, ".txt"},
			{"long_trace_id", "logs", longA, "span", "event", ".txt"},
			{"long_span_id", "logs", "trace", longA, "event", ".txt"},
			{"long_single_segment_prefix", longA, "trace", "span", "event", ".txt"},
			{"long_multi_segment_prefix", longMultiSegPrefix, "trace", "span", "event", ".txt"},
			{"all_long_ascii", longMultiSegPrefix, longA, longA, longA, ".json"},
			{"all_long_utf8", longUTF8, longUTF8, longUTF8, longUTF8, ".bin"},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				got := gcsObjectPath(tc.prefix, ts, tc.traceID, tc.spanID, tc.eventID, 0, 0, tc.ext)
				if len(got) > wantObjectLimit {
					t.Errorf("len(gcsObjectPath) = %d, exceeds %d: %q", len(got), wantObjectLimit, got)
				}
				if !strings.HasSuffix(got, tc.ext) {
					t.Errorf("gcsObjectPath = %q, want suffix %q", got, tc.ext)
				}
			})
		}

		pA := gcsObjectPath("logs", ts, "trace", "span", longA, 0, 0, ".txt")
		pB := gcsObjectPath("logs", ts, "trace", "span", longB, 0, 0, ".txt")
		if pA == pB {
			t.Errorf("distinct 1100-byte event IDs sharing 1000-byte prefix collided on %q", pA)
		}

		// With plugin-length hex identifiers (32, 16, and 32 bytes), the longest
		// int ordinals, and a 17-byte extension from extensionForMIME, a 750-byte
		// prefix stays at 905 bytes without hitting the whole-name cap. A direct
		// call with 128-byte identifiers hits the 1024-byte cap while preserving
		// a safe extension.
		safeExt := ".abcdefghijklmnop"
		maxPrefix := strings.Repeat("p", 750)
		gotPluginMax := gcsObjectPath(maxPrefix, ts, strings.Repeat("a", 32), strings.Repeat("b", 16), strings.Repeat("c", 32), math.MinInt, math.MinInt, safeExt)
		if len(gotPluginMax) != 905 || !strings.HasPrefix(gotPluginMax, maxPrefix+"/") || !strings.HasSuffix(gotPluginMax, safeExt) {
			t.Errorf("gcsObjectPath with 750-byte prefix and plugin hex IDs = %q (%d bytes), want 905 bytes starting with prefix and ending in %q", gotPluginMax, len(gotPluginMax), safeExt)
		}
		if got := gcsObjectPath(longA, ts, longA, longA, longA, math.MinInt, math.MinInt, safeExt); len(got) != wantObjectLimit || !strings.HasSuffix(got, safeExt) {
			t.Errorf("gcsObjectPath with the longest direct-call parts and a safe extension = %q (%d bytes), want %d bytes ending in %q", got, len(got), wantObjectLimit, safeExt)
		}
		// Focused exact-1,024-byte cap check across safe, empty, and oversized
		// extensions so changing maxGCSObjectNameBytes away from 1,024 fails
		// immediately on both truncation branches.
		trace128 := strings.Repeat("t", 128)
		span128 := strings.Repeat("s", 128)
		event128 := strings.Repeat("e", 128)
		for _, ext := range []string{".txt", safeExt, "", strings.Repeat("x", 1100), "." + strings.Repeat("x", 1100)} {
			got := gcsObjectPath(maxPrefix, ts, trace128, span128, event128, 0, 0, ext)
			if len(got) != wantObjectLimit {
				t.Errorf("ext len %d: len(gcsObjectPath) = %d, want exact %d", len(ext), len(got), wantObjectLimit)
			}
		}
		// Two safe-extension direct-call names that overflow 1,024 bytes and
		// differ only after the truncation cut must preserve their safe
		// extension, stay at 1,024 bytes, remain distinct, and embed the full
		// SHA-256 digest of each entire uncapped object name.
		event128A := strings.Repeat("e", 127) + "a"
		event128B := strings.Repeat("e", 127) + "b"
		for _, ext := range []string{".txt", safeExt} {
			uncappedA := fmt.Sprintf("%s/2026-03-04/%s/%s_%s_c0_p0%s", maxPrefix, trace128, span128, event128A, ext)
			uncappedB := fmt.Sprintf("%s/2026-03-04/%s/%s_%s_c0_p0%s", maxPrefix, trace128, span128, event128B, ext)
			cut := wantObjectLimit - wantHashSuffixLen - len(ext)
			if uncappedA[:cut] != uncappedB[:cut] {
				t.Fatalf("ext %q: uncapped inputs already differ before cut %d", ext, cut)
			}
			sumA := sha256.Sum256([]byte(uncappedA))
			sumB := sha256.Sum256([]byte(uncappedB))
			wantA := fmt.Sprintf("%s~%x%s", uncappedA[:cut], sumA, ext)
			wantB := fmt.Sprintf("%s~%x%s", uncappedB[:cut], sumB, ext)
			gotA := gcsObjectPath(maxPrefix, ts, trace128, span128, event128A, 0, 0, ext)
			gotB := gcsObjectPath(maxPrefix, ts, trace128, span128, event128B, 0, 0, ext)
			if len(gotA) != wantObjectLimit || !strings.HasSuffix(gotA, ext) {
				t.Errorf("ext %q: gotA = %q (%d bytes), want %d bytes ending in %q", ext, gotA, len(gotA), wantObjectLimit, ext)
			}
			if len(gotB) != wantObjectLimit || !strings.HasSuffix(gotB, ext) {
				t.Errorf("ext %q: gotB = %q (%d bytes), want %d bytes ending in %q", ext, gotB, len(gotB), wantObjectLimit, ext)
			}
			if gotA == gotB {
				t.Errorf("ext %q: distinct safe-extension overflow names differing after cut collided on %q", ext, gotA)
			}
			if gotA != wantA {
				t.Errorf("ext %q: gotA = %q, want %q (full SHA-256 of uncapped name)", ext, gotA, wantA)
			}
			if gotB != wantB {
				t.Errorf("ext %q: gotB = %q, want %q (full SHA-256 of uncapped name)", ext, gotB, wantB)
			}
		}
		// Only an oversized extension passed in directly reaches the cap on the
		// whole name. The capped name ends in the hash of the whole uncapped
		// name, so names that differ only after the cut stay apart.
		for _, ext := range []string{"." + strings.Repeat("x", 1100), "." + strings.Repeat("x", 1099) + "y"} {
			uncapped := "logs/2026-03-04/trace/span_event_c0_p0" + ext
			want := uncapped[:wantObjectLimit-wantHashSuffixLen] + pathSegmentHashSuffix(uncapped)
			if got := gcsObjectPath("logs", ts, "trace", "span", "event", 0, 0, ext); got != want || len(got) != wantObjectLimit {
				t.Errorf("gcsObjectPath with a %d-byte extension = %q (%d bytes), want %q (%d bytes)", len(ext), got, len(got), want, wantObjectLimit)
			}
		}
	})

	t.Run("an extension cannot make a name equal to a hashed one", func(t *testing.T) {
		// '~' marks a name that was rewritten to end in the hash of the
		// whole name, so an extension passed in directly must not be able
		// to make a name equal to a rewritten one.
		pathWith := func(ext string) string {
			return gcsObjectPath("logs", ts, "trace", "span", "event", 0, 0, ext)
		}
		base := pathWith("")

		// An oversized extension makes p1, which ends in the hash of the
		// whole uncapped name. What follows base in p1 is an extension of
		// its own, and the name that it makes is as long as p1: if a name
		// that fits were returned as is, that would be p1 itself.
		p1 := pathWith("." + strings.Repeat("x", 1100))
		if !strings.HasPrefix(p1, base) || len(p1) != wantObjectLimit {
			t.Fatalf("name with an oversized extension = %q (%d bytes), want %d bytes starting with %q", p1, len(p1), wantObjectLimit, base)
		}
		ext2 := strings.TrimPrefix(p1, base)
		p2 := pathWith(ext2)
		if p2 == p1 {
			t.Errorf("name with extension %q = %q, the name that an oversized extension makes", ext2, p2)
		}
		if len(p2) > wantObjectLimit {
			t.Errorf("len(name with extension %q) = %d, exceeds %d", ext2, len(p2), wantObjectLimit)
		}
		if !utf8.ValidString(p2) {
			t.Errorf("name with extension %q = %q, want valid UTF-8", ext2, p2)
		}
		if want := p1[:wantObjectLimit-wantHashSuffixLen] + pathSegmentHashSuffix(p1); p2 != want {
			t.Errorf("name with extension %q = %q, want %q: the hash of the whole name p1", ext2, p2, want)
		}

		// A name that fits is hashed too if its extension is not safe, and
		// the extension that the hashed name ends in makes another name
		// again. '~' and '/' in the extension read as '_' in the name, and
		// the hash is that of the extension as it was.
		for _, tc := range []struct{ ext, readable string }{
			{".a~b", ".a_b"},
			{"~", "_"},
			{pathSegmentHashSuffix(base), "_" + strings.TrimPrefix(pathSegmentHashSuffix(base), "~")},
			{".a b", ".a b"},
			{"png", "png"},
			{".png/..", ".png_.."},
			{".", "."},
		} {
			got := pathWith(tc.ext)
			if want := base + tc.readable + pathSegmentHashSuffix(base+tc.ext); got != want {
				t.Errorf("name with extension %q = %q, want %q", tc.ext, got, want)
			}
			if again := pathWith(strings.TrimPrefix(got, base)); again == got {
				t.Errorf("names with extensions %q and %q both = %q", tc.ext, strings.TrimPrefix(got, base), got)
			}
		}
		// An empty or safe extension is kept as is.
		for _, ext := range []string{"", ".txt", ".abcdefghijklmnop"} {
			if got, want := pathWith(ext), base+ext; got != want {
				t.Errorf("name with extension %q = %q, want %q", ext, got, want)
			}
		}
	})

	t.Run("an invalid UTF-8 extension gives a valid name", func(t *testing.T) {
		// The prefix and the ids are valid UTF-8, so the extension is the one
		// part of a direct call that can make a name invalid. The readable
		// part of the name has each run of invalid bytes replaced with '_',
		// and the hash is that of the name with the extension as it was, so
		// extensions that share a readable part are told apart by the hash.
		pathWith := func(ext string) string {
			return gcsObjectPath("logs", ts, "trace", "span", "event", 0, 0, ext)
		}
		base := pathWith("")

		names := map[string]string{}
		check := func(desc, name string) {
			t.Helper()
			if !utf8.ValidString(name) {
				t.Errorf("%s: name = %q, want valid UTF-8", desc, name)
			}
			if len(name) > wantObjectLimit {
				t.Errorf("%s: len(name) = %d, exceeds %d", desc, len(name), wantObjectLimit)
			}
			if prev, ok := names[name]; ok {
				t.Errorf("%s and %s both give the name %q", prev, desc, name)
			}
			names[name] = desc
		}

		for _, tc := range []struct {
			name string
			ext  string
			// readable is what follows base ahead of the hash suffix.
			readable string
		}{
			{"0xff after the dot", string([]byte{'.', 0xff}), "._"},
			{"0xfe after the dot", string([]byte{'.', 0xfe}), "._"},
			{"the replacement as it is", "._", "._"},
			{"0xff alone", string([]byte{0xff}), "_"},
			{"the start of a two-byte rune", ".\xc3", "._"},
			{"two bytes of a three-byte rune", ".\xe6\x97", "._"},
			{"three bytes of a four-byte rune", ".\xf0\x9f\x98", "._"},
			{"valid runes and an invalid byte", ".日本語\xff", ".日本語_"},
			{"a long run of invalid bytes", "." + strings.Repeat("\xff", 1100), "._"},
		} {
			got := pathWith(tc.ext)
			if want := base + tc.readable + pathSegmentHashSuffix(base+tc.ext); got != want {
				t.Errorf("%s: name = %q, want %q", tc.name, got, want)
			}
			check(tc.name, got)
		}

		// An oversized extension is cut in the readable part, which is ASCII
		// here, and the hash is still that of the whole raw name.
		long := "." + strings.Repeat("\xffa", 600)
		readable := base + "." + strings.Repeat("_a", 600)
		want := readable[:wantObjectLimit-wantHashSuffixLen] + pathSegmentHashSuffix(base+long)
		got := pathWith(long)
		if got != want {
			t.Errorf("name with an oversized extension of ASCII and invalid bytes = %q, want %q", got, want)
		}
		check("an oversized extension of ASCII and invalid bytes", got)

		// Multi-byte runes between invalid bytes, cut by the byte limit.
		for _, tc := range []struct{ name, ext string }{
			{"three-byte runes between invalid bytes", "." + strings.Repeat("日\xff", 400)},
			{"four-byte runes between invalid bytes", "." + strings.Repeat("😀\xff", 300)},
		} {
			check(tc.name, pathWith(tc.ext))
		}
	})

	t.Run("an unsafe extension cannot put a line break or control character in the name", func(t *testing.T) {
		// Cloud Storage rejects CR and LF in an object name. The prefix and the
		// ids never hold a control character, so the extension of a direct
		// call is the one part that can. The readable part of the name has
		// each such rune, each other rune that cleanGCSPrefix replaces, and
		// each '/', '\', and '~', replaced with '_', and the hash is that of
		// the name with the extension as it was, so extensions that read the
		// same are told apart by the hash.
		pathWith := func(ext string) string {
			return gcsObjectPath("logs", ts, "trace", "span", "event", 0, 0, ext)
		}
		base := pathWith("")

		names := map[string]string{}
		check := func(desc, name string) {
			t.Helper()
			if !utf8.ValidString(name) {
				t.Errorf("%s: name = %q, want valid UTF-8", desc, name)
			}
			if len(name) > wantObjectLimit {
				t.Errorf("%s: len(name) = %d, exceeds %d", desc, len(name), wantObjectLimit)
			}
			for i, r := range name {
				if unicode.IsControl(r) || r == '\u2028' || r == '\u2029' || r == '\\' {
					t.Errorf("%s: name has %U at byte %d, want no control character, line or paragraph separator or backslash", desc, r, i)
					break
				}
			}
			if prev, ok := names[name]; ok {
				t.Errorf("%s and %s both give the name %q", prev, desc, name)
			}
			names[name] = desc
		}

		for _, tc := range []struct {
			name string
			ext  string
			// readable is what follows base ahead of the hash suffix.
			readable string
		}{
			{"a carriage return", ".png\r", ".png_"},
			{"a line feed", ".png\n", ".png_"},
			{"a carriage return and a line feed", ".png\r\n", ".png__"},
			{"a line feed and a carriage return", ".png\n\r", ".png__"},
			{"a line feed alone", "\n", "_"},
			{"a line feed in the middle", ".p\nng", ".p_ng"},
			{"NUL", ".png\x00", ".png_"},
			{"a tab", ".png\t", ".png_"},
			{"the last C0 control character", ".png\x1f", ".png_"},
			{"DEL", ".png\x7f", ".png_"},
			{"the first C1 control character", ".png\u0080", ".png_"},
			{"next line", ".png\u0085", ".png_"},
			{"the last C1 control character", ".png\u009f", ".png_"},
			{"line separator", ".png\u2028", ".png_"},
			{"paragraph separator", ".png\u2029", ".png_"},
			{"a backslash", `.png\`, ".png_"},
			{"a tilde", ".png~", ".png_"},
			{"a slash", ".png/", ".png_"},
			{"a slash before a dot-dot", ".png/..", ".png_.."},
			{"a slash alone", "/", "_"},
			{"line breaks around an invalid byte", ".png\r\xff\n", ".png___"},
			{"the replacement as it is", ".png_", ".png_"},
			{"two replacements as they are", ".png__", ".png__"},
			{"a literal U+FFFD", ".png\uFFFD", ".png\uFFFD"},
			{"spaces and Unicode", ".a b日本é", ".a b日本é"},
		} {
			got := pathWith(tc.ext)
			if want := base + tc.readable + pathSegmentHashSuffix(base+tc.ext); got != want {
				t.Errorf("%s: name = %q, want %q", tc.name, got, want)
			}
			check(tc.name, got)
		}

		// An oversized extension is cut in the readable part, after the
		// replacements, and the hash is still that of the whole raw name.
		for _, tc := range []struct {
			name string
			ext  string
			// readable is the name before the cut: the base name and the
			// extension as the name shows it.
			readable string
		}{
			{"line breaks", "." + strings.Repeat("\r\n", 600), base + "." + strings.Repeat("__", 600)},
			{"line separators between letters", "." + strings.Repeat("\u2028a", 500), base + "." + strings.Repeat("_a", 500)},
		} {
			want := tc.readable[:wantObjectLimit-wantHashSuffixLen] + pathSegmentHashSuffix(base+tc.ext)
			got := pathWith(tc.ext)
			if got != want {
				t.Errorf("name with an oversized extension of %s = %q, want %q", tc.name, got, want)
			}
			check("an oversized extension of "+tc.name, got)
		}

		// The cut backs up to a rune boundary of the readable part: 959 bytes
		// fall inside the 231st three-byte rune, so the name keeps 230 of them.
		multi := strings.Repeat("日\r", 400)
		wantMulti := base + strings.Repeat("日_", 230) + pathSegmentHashSuffix(base+multi)
		gotMulti := pathWith(multi)
		if gotMulti != wantMulti {
			t.Errorf("name with an oversized extension of three-byte runes and carriage returns = %q, want %q", gotMulti, wantMulti)
		}
		check("an oversized extension of three-byte runes and carriage returns", gotMulti)
	})

	t.Run("prefix sanitization and reserved paths", func(t *testing.T) {
		prefixes := []struct {
			prefix   string
			wantLead string
		}{
			{"tenant=acme", "tenant=acme/2026-03-04/"},
			{"logs v1", "logs v1/2026-03-04/"},
			{"logs/テナント", "logs/テナント/2026-03-04/"},
			{"logs/.../tenant", "logs/.../tenant/2026-03-04/"},
			{"a/.../b", "a/.../b/2026-03-04/"},
			{"../x", "../x/2026-03-04/"},
			{"a/../b", "a/../b/2026-03-04/"},
			{"a//b", "a//b/2026-03-04/"},
			{".", "./2026-03-04/"},
			{"..", "../2026-03-04/"},
			{"a/./b", "a/./b/2026-03-04/"},
			{`team\prod/v1`, `team\prod/v1/2026-03-04/`},
			{`~team\prod/v~1`, `_team\prod/v_1` + pathSegmentHashSuffix(`~team\prod/v~1`) + `/2026-03-04/`},
			{".well-known/acme-challenge-x", ".well-known/acme-challenge-x/2026-03-04/"},
			{strings.Repeat("a", 750), strings.Repeat("a", 750) + "/2026-03-04/"},
			{strings.Repeat("a", 751), strings.Repeat("a", 685) + pathSegmentHashSuffix(strings.Repeat("a", 751)) + "/2026-03-04/"},
			{".well-known/acme-challenge/x", "_well-known/acme-challenge/x" + pathSegmentHashSuffix(".well-known/acme-challenge/x") + "/2026-03-04/"},
			{"/.well-known/ACME-CHALLENGE/x", "_well-known/ACME-CHALLENGE/x" + pathSegmentHashSuffix(".well-known/ACME-CHALLENGE/x") + "/2026-03-04/"},
			{"logs\ninjected/part\r\n2\x00", "logs_injected/part__2_" + pathSegmentHashSuffix("logs\ninjected/part\r\n2\x00") + "/2026-03-04/"},
			{"logs\u2028sep\u2029", "logs_sep_" + pathSegmentHashSuffix("logs\u2028sep\u2029") + "/2026-03-04/"},
			{"logs/\xff", "logs/_" + pathSegmentHashSuffix("logs/\xff") + "/2026-03-04/"},
		}
		for _, tc := range prefixes {
			got := gcsObjectPath(tc.prefix, ts, "trace", "span", "event", 0, 0, ".txt")
			if tc.wantLead != "" && !strings.HasPrefix(got, tc.wantLead) {
				t.Errorf("gcsObjectPath(prefix=%q) = %q, want prefix %q", tc.prefix, got, tc.wantLead)
			}
			if !utf8.ValidString(got) || len(got) > wantObjectLimit {
				t.Errorf("gcsObjectPath(prefix=%q) = %q (%d bytes), want valid UTF-8 <= %d bytes", tc.prefix, got, len(got), wantObjectLimit)
			}
			if seg0, rest, ok := strings.Cut(got, "/"); ok && strings.EqualFold(seg0, ".well-known") {
				if seg1, _, _ := strings.Cut(rest, "/"); strings.EqualFold(seg1, "acme-challenge") {
					t.Errorf("gcsObjectPath(prefix=%q) = %q begins with reserved .well-known/acme-challenge segment pair", tc.prefix, got)
				}
			}
			for _, r := range got {
				if isUnsafeGCSPrefixRune(r) {
					t.Errorf("gcsObjectPath(prefix=%q) = %q contains unsafe rune %U", tc.prefix, got, r)
				}
			}
		}

		// Distinct configured prefixes (after leading/trailing '/' trim), including
		// rewritten prefixes and their generated "<base>~<sha256>" outputs reused as
		// literal Config.GCSPathPrefix values, must never collapse onto the same
		// object path or lose create-only uploads in a shared bucket.
		overLimitRaw := strings.Repeat("a", maxGCSPrefixBytes+1)
		distinctPrefixes := []string{
			"",
			".",
			"..",
			"./..",
			"...",
			"logs/tenant",
			"logs/.../tenant",
			"logs/../tenant",
			"logs/./tenant",
			"logs//tenant",
			`team\prod`,
			`~team\prod`,
			cleanGCSPrefix(`~team\prod`),
			`_team\prod`,
			".well-known/acme-challenge-x",
			".well-known/acme-challenge/x",
			cleanGCSPrefix(".well-known/acme-challenge/x"),
			"_well-known/acme-challenge/x",
			"logs\tx",
			"logs\nx",
			cleanGCSPrefix("logs\nx"),
			"logs_x",
			"\xff",
			cleanGCSPrefix("\xff"),
			"_",
			overLimitRaw,
			cleanGCSPrefix(overLimitRaw),
		}
		seenPrefixPaths := make(map[string]string, len(distinctPrefixes))
		type storedCheck struct {
			prefix      string
			objectPath  string
			wantPayload string
		}
		storedChecks := make([]storedCheck, 0, len(distinctPrefixes))
		fs := newFakeStorage(t)
		offloader, err := newGCSOffloader(context.Background(), "test-bucket", fs.clientOptions()...)
		if err != nil {
			t.Fatalf("newGCSOffloader() error: %v", err)
		}
		t.Cleanup(func() { _ = offloader.Close() })

		for i, p := range distinctPrefixes {
			got := gcsObjectPath(p, ts, "trace", "span", "event", 0, 0, ".txt")
			if prev, exists := seenPrefixPaths[got]; exists {
				t.Errorf("distinct prefixes %q and %q collided on GCS object path %q", prev, p, got)
			}
			seenPrefixPaths[got] = p

			wantPayload := fmt.Sprintf("prefix-payload-%d-%s", i, p)
			bpCfg := DefaultConfig()
			bpCfg.GCSOffloader = offloader
			bpCfg.GCSPathPrefix = p
			bpCfg.ConnectionID = "us.conn"
			bpCfg.Logger = &recordingLogger{}
			bp, err := NewBatchProcessor(context.Background(), nil, "test_stream", bpCfg)
			if err != nil {
				t.Fatalf("NewBatchProcessor(prefix=%q) error: %v", p, err)
			}
			opts := partsOptions{maxLen: 1000, offload: true, inlineLimit: 5}
			row := queuedRow(opts, &genai.Content{Role: "user", Parts: []*genai.Part{{Text: wantPayload}}})
			row[timestampField] = ts
			row["trace_id"] = "trace"
			row["span_id"] = "span"
			row[eventIDField] = "event"
			bp.resolveUploads(context.Background(), false, []map[string]any{row})
			bp.Close()

			uri, _ := row[columnContentParts].([]map[string]any)[0]["uri"].(string)
			wantURI := "gs://test-bucket/" + got
			if uri != wantURI {
				t.Errorf("prefix %q uploaded to %q, want %q", p, uri, wantURI)
			}
			objPath := strings.TrimPrefix(uri, "gs://test-bucket/")
			obj, ok := fs.object("test-bucket", objPath)
			if !ok || string(obj.data) != wantPayload {
				t.Errorf("prefix %q stored (%v, %q), want %q at %q", p, ok, obj.data, wantPayload, uri)
			}
			storedChecks = append(storedChecks, storedCheck{prefix: p, objectPath: objPath, wantPayload: wantPayload})
		}
		for _, sc := range storedChecks {
			obj, ok := fs.object("test-bucket", sc.objectPath)
			if !ok || string(obj.data) != sc.wantPayload {
				t.Errorf("after all uploads, prefix %q object %q holds (%v, %q), want %q", sc.prefix, sc.objectPath, ok, obj.data, sc.wantPayload)
			}
		}
	})
}

func TestGCSPathPrefix_RewriteWarningAndDisjointNamespace(t *testing.T) {
	countWarnings := func(l *recordingLogger) int {
		l.mu.Lock()
		defer l.mu.Unlock()
		n := 0
		for _, line := range l.lines {
			if strings.Contains(line, "WARNING: Config.GCSPathPrefix") {
				n++
			}
		}
		return n
	}

	t.Run("valid prefixes log no warning and pass verbatim to custom offloader", func(t *testing.T) {
		for _, prefix := range []string{
			"",
			"/logs/tenant/",
			`logs\prod`,
			"a//b",
			"a/./b",
			"../a/../b",
			".well-known/acme-challenge-x",
			strings.Repeat("a", maxGCSPrefixBytes),
		} {
			off := &fakeOffloader{}
			log := &recordingLogger{}
			p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
				c.GCSPathPrefix = prefix
				c.MaxInlineTextSize = 5
				c.Logger = log
			})
			if got := countWarnings(log); got != 0 {
				t.Errorf("prefix %q logged %d warnings (%s), want 0", prefix, got, log.String())
			}

			row := logRequest(t, p, requests, mCtx, &model.LLMRequest{
				Model:    "gemini-2.5-flash",
				Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "offloaded text payload"}}}},
			})
			uploads := off.uploads()
			if len(uploads) != 1 {
				t.Fatalf("prefix %q: len(uploads) = %d, want 1", prefix, len(uploads))
			}
			wantLead := rowDate(row) + "/"
			if trimmed := strings.Trim(prefix, "/"); trimmed != "" {
				wantLead = trimmed + "/" + wantLead
			}
			if !strings.HasPrefix(uploads[0].Path, wantLead) {
				t.Errorf("prefix %q: GCSUploadRequest.Path = %q, want prefix %q", prefix, uploads[0].Path, wantLead)
			}
		}
	})

	t.Run("rewritten prefixes log constant privacy-safe warning and stay disjoint when reused", func(t *testing.T) {
		const secretSentinel = "PrivateCustomerSecretMarker98765"
		for _, tc := range []struct {
			name       string
			prefix     string
			wantReason string
		}{
			{"reserved acme-challenge", ".well-known/acme-challenge/" + secretSentinel, "starts with reserved .well-known/acme-challenge prefix"},
			{"control character", "tenant-" + secretSentinel + "\ninjected", "contains control or line-separator character"},
			{"reserved tilde", "tenant-" + secretSentinel + "~v1", "contains reserved '~'"},
			{"invalid UTF-8", "tenant-" + secretSentinel + "/\xff", "contains invalid UTF-8"},
			{"over 750 bytes", secretSentinel + strings.Repeat("a", maxGCSPrefixBytes+1-len(secretSentinel)), "exceeds 750 bytes"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				trimmed := strings.Trim(tc.prefix, "/")
				wantEffective := cleanGCSPrefix(tc.prefix)
				off := &fakeOffloader{}
				log := &recordingLogger{}
				p, requests, mCtx := offloadingPlugin(t, off, func(c *Config) {
					c.GCSPathPrefix = tc.prefix
					c.MaxInlineTextSize = 5
					c.Logger = log
				})
				if got := countWarnings(log); got != 1 {
					t.Fatalf("logged %d warnings (%s), want exactly 1", got, log.String())
				}
				if log.contains(secretSentinel) {
					t.Errorf("warning %q leaked private sentinel %q", log.String(), secretSentinel)
				}
				if log.contains(tc.prefix) || log.contains(wantEffective) {
					t.Errorf("warning %q logged raw or rewritten prefix text", log.String())
				}
				digestHex := strings.TrimPrefix(pathSegmentHashSuffix(trimmed), "~")
				if log.contains(digestHex) {
					t.Errorf("warning %q logged prefix SHA-256 digest %q", log.String(), digestHex)
				}
				wantCounts := fmt.Sprintf("(%d raw bytes, %d trimmed bytes) was rewritten (%d effective bytes)", len(tc.prefix), len(trimmed), len(wantEffective))
				if !log.contains(wantCounts) {
					t.Errorf("warning %q missing byte counts %q", log.String(), wantCounts)
				}
				if !log.contains(tc.wantReason) {
					t.Errorf("warning %q missing reason %q", log.String(), tc.wantReason)
				}

				row := logRequest(t, p, requests, mCtx, &model.LLMRequest{
					Model:    "gemini-2.5-flash",
					Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "offloaded text payload"}}}},
				})
				uploads := off.uploads()
				if len(uploads) != 1 {
					t.Fatalf("len(uploads) = %d, want 1", len(uploads))
				}
				wantLead := wantEffective + "/" + rowDate(row) + "/"
				if !strings.HasPrefix(uploads[0].Path, wantLead) {
					t.Errorf("GCSUploadRequest.Path = %q, want prefix %q", uploads[0].Path, wantLead)
				}

				// Passing a generated "<base>~<sha256>" output back into
				// Config.GCSPathPrefix must rewrite it again (never alias the
				// original rewritten prefix) and log a privacy-safe '~' warning.
				wantReusedEffective := cleanGCSPrefix(wantEffective)
				if wantReusedEffective == wantEffective {
					t.Fatalf("cleanGCSPrefix(%q) = %q, want disjoint rewritten prefix", wantEffective, wantReusedEffective)
				}
				offReuse := &fakeOffloader{}
				logReuse := &recordingLogger{}
				pReuse, reqReuse, mCtxReuse := offloadingPlugin(t, offReuse, func(c *Config) {
					c.GCSPathPrefix = wantEffective
					c.MaxInlineTextSize = 5
					c.Logger = logReuse
				})
				if got := countWarnings(logReuse); got != 1 {
					t.Errorf("reusing effective prefix logged %d warnings (%s), want 1", got, logReuse.String())
				}
				if logReuse.contains(secretSentinel) || logReuse.contains(wantEffective) || logReuse.contains(wantReusedEffective) {
					t.Errorf("reuse warning %q leaked prefix text or sentinel", logReuse.String())
				}
				if !logReuse.contains("contains reserved '~'") {
					t.Errorf("reuse warning %q missing reserved '~' reason", logReuse.String())
				}
				rowReuse := logRequest(t, pReuse, reqReuse, mCtxReuse, &model.LLMRequest{
					Model:    "gemini-2.5-flash",
					Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "offloaded text payload"}}}},
				})
				uploadsReuse := offReuse.uploads()
				wantReuseLead := wantReusedEffective + "/" + rowDate(rowReuse) + "/"
				if len(uploadsReuse) != 1 || !strings.HasPrefix(uploadsReuse[0].Path, wantReuseLead) {
					t.Errorf("reusing effective prefix uploaded %+v, want prefix %q", uploadsReuse, wantReuseLead)
				}
				if strings.HasPrefix(uploadsReuse[0].Path, wantEffective+"/") {
					t.Errorf("reused effective prefix aliased original rewritten prefix %q in path %q", wantEffective, uploadsReuse[0].Path)
				}
			})
		}
	})

	t.Run("direct NewBatchProcessor warns once without leaking prefix and disabled plugin stays quiet", func(t *testing.T) {
		ctx := context.Background()
		lis, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			t.Fatalf("net.Listen: %v", err)
		}
		gSrv := grpc.NewServer()
		storagepb.RegisterBigQueryWriteServer(gSrv, &fakeBigQueryWriteServer{requests: make(chan *storagepb.AppendRowsRequest, 10)})
		go func() { _ = gSrv.Serve(lis) }()
		t.Cleanup(gSrv.Stop)

		conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			t.Fatalf("grpc.NewClient: %v", err)
		}
		t.Cleanup(func() { _ = conn.Close() })

		wc, err := bqstorage.NewBigQueryWriteClient(ctx, option.WithGRPCConn(conn))
		if err != nil {
			t.Fatalf("NewBigQueryWriteClient: %v", err)
		}
		t.Cleanup(func() { _ = wc.Close() })

		const directSentinel = "DirectCustomerSecret42"
		warnLog := &recordingLogger{}
		cfgWarn := DefaultConfig()
		cfgWarn.ProjectID = "test-project"
		cfgWarn.DatasetID = "test_dataset"
		cfgWarn.TableName = "test-table"
		cfgWarn.GCSPathPrefix = "/.well-known/acme-challenge/" + directSentinel + "/"
		cfgWarn.Logger = warnLog
		bpWarn, err := NewBatchProcessor(ctx, wc, "", cfgWarn)
		if err != nil {
			t.Fatalf("NewBatchProcessor(rewrite) error: %v", err)
		}
		bpWarn.Close()
		wantWarnEffective := cleanGCSPrefix(cfgWarn.GCSPathPrefix)
		wantDirectCounts := fmt.Sprintf("(%d raw bytes, %d trimmed bytes) was rewritten (%d effective bytes)",
			len(cfgWarn.GCSPathPrefix), len(strings.Trim(cfgWarn.GCSPathPrefix, "/")), len(wantWarnEffective))
		if got := countWarnings(warnLog); got != 1 {
			t.Errorf("direct NewBatchProcessor(rewrite) warnings = %d (%s), want 1", got, warnLog.String())
		}
		if warnLog.contains(directSentinel) || warnLog.contains(wantWarnEffective) {
			t.Errorf("direct NewBatchProcessor(rewrite) warning %q leaked prefix text", warnLog.String())
		}
		if !warnLog.contains(wantDirectCounts) || !warnLog.contains("starts with reserved .well-known/acme-challenge prefix") {
			t.Errorf("direct NewBatchProcessor(rewrite) warning %q missing counts %q or reason", warnLog.String(), wantDirectCounts)
		}

		okLog := &recordingLogger{}
		cfgOK := cfgWarn
		cfgOK.GCSPathPrefix = `team\logs`
		cfgOK.Logger = okLog
		bpOK, err := NewBatchProcessor(ctx, wc, "", cfgOK)
		if err != nil {
			t.Fatalf("NewBatchProcessor(valid) error: %v", err)
		}
		bpOK.Close()
		if got := countWarnings(okLog); got != 0 {
			t.Errorf("direct NewBatchProcessor(valid) warnings = %d (%s), want 0", got, okLog.String())
		}

		disabledLog := &recordingLogger{}
		cfgDisabled := DefaultConfig()
		cfgDisabled.Enabled = false
		cfgDisabled.GCSPathPrefix = "logs\ninjected-" + directSentinel
		cfgDisabled.Logger = disabledLog
		pDisabled, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfgDisabled, nil, nil)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients(disabled) error: %v", err)
		}
		_ = pDisabled.Close()
		if got := countWarnings(disabledLog); got != 0 {
			t.Errorf("disabled plugin warnings = %d (%s), want 0", got, disabledLog.String())
		}
	})
}

func TestBatchProcessor_ObjectPathCachesEffectivePrefix(t *testing.T) {
	ts := time.Date(2026, 3, 4, 5, 6, 7, 0, time.UTC)
	row := map[string]any{
		timestampField: ts,
		"trace_id":     "trace-1",
		"span_id":      "span-1",
		eventIDField:   "event-1",
	}
	upload := pendingUpload{contentOrd: 0, partIdx: 0, mimeType: textUploadContentType}

	for _, tc := range []struct {
		name   string
		prefix string
	}{
		{"4MiB_raw_prefix", "tenant/" + strings.Repeat("a", 4<<20)},
		{"control_char_prefix", "logs\nx"},
		{"slash_only_4MiB_prefix", strings.Repeat("/", 4<<20)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			log := &recordingLogger{}
			cfg := DefaultConfig()
			cfg.GCSPathPrefix = tc.prefix
			cfg.Logger = log

			bp, err := NewBatchProcessor(context.Background(), nil, "test_stream", cfg)
			if err != nil {
				t.Fatalf("NewBatchProcessor() error: %v", err)
			}
			t.Cleanup(bp.Close)

			if bp.config.GCSPathPrefix != tc.prefix {
				t.Fatalf("len(bp.config.GCSPathPrefix) = %d, want raw %d-byte prefix preserved unmutated", len(bp.config.GCSPathPrefix), len(tc.prefix))
			}

			wantClean := cleanGCSPrefix(tc.prefix)
			wantPath := gcsObjectPath(tc.prefix, ts, "trace-1", "span-1", "event-1", 0, 0, ".txt")
			doubleNormalizedPath := gcsObjectPath(wantClean, ts, "trace-1", "span-1", "event-1", 0, 0, ".txt")

			for i := range 3 {
				got := bp.objectPath(row, upload)
				if got != wantPath {
					t.Fatalf("call %d: bp.objectPath() = %q, want %q", i, got, wantPath)
				}
				if wantClean != "" && got == doubleNormalizedPath {
					t.Fatalf("call %d: bp.objectPath() double-normalized effective prefix to %q", i, got)
				}
			}

			var sink string
			sink = bp.objectPath(row, upload)
			runtime.GC()
			var m0, m1 runtime.MemStats
			runtime.ReadMemStats(&m0)
			const runs = 200
			for range runs {
				sink = bp.objectPath(row, upload)
			}
			runtime.ReadMemStats(&m1)
			_ = sink

			bytesPerCall := (m1.TotalAlloc - m0.TotalAlloc) / runs
			if bytesPerCall >= 16*1024 {
				t.Errorf("bp.objectPath() allocated %d bytes/call for %d-byte raw prefix, want < 16384 bytes/call", bytesPerCall, len(tc.prefix))
			}
			if allocs := testing.AllocsPerRun(100, func() { sink = bp.objectPath(row, upload) }); allocs > 20 {
				t.Errorf("bp.objectPath() allocs/op = %v, want <= 20", allocs)
			}
		})
	}
}

func BenchmarkBatchProcessorObjectPath(b *testing.B) {
	ts := time.Date(2026, 3, 4, 5, 6, 7, 0, time.UTC)
	row := map[string]any{
		timestampField: ts,
		"trace_id":     "trace-1",
		"span_id":      "span-1",
		eventIDField:   "event-1",
	}
	upload := pendingUpload{contentOrd: 0, partIdx: 0, mimeType: textUploadContentType}

	for _, tc := range []struct {
		name   string
		prefix string
	}{
		{"751B", strings.Repeat("a", 751)},
		{"4MiB", strings.Repeat("a", 4<<20)},
	} {
		b.Run(tc.name, func(b *testing.B) {
			cfg := DefaultConfig()
			cfg.GCSPathPrefix = tc.prefix
			cfg.Logger = &recordingLogger{}
			bp, err := NewBatchProcessor(context.Background(), nil, "test_stream", cfg)
			if err != nil {
				b.Fatalf("NewBatchProcessor() error: %v", err)
			}
			b.Cleanup(bp.Close)

			b.ReportAllocs()
			b.ResetTimer()
			var sink string
			for i := 0; i < b.N; i++ {
				sink = bp.objectPath(row, upload)
			}
			_ = sink
		})
	}
}
