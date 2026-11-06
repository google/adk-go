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
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	bq "cloud.google.com/go/bigquery"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"

	baseplugin "google.golang.org/adk/v2/plugin"
)

func TestNotifySchemaCallbacks(t *testing.T) {
	t.Run("nil callbacks are no-ops", func(t *testing.T) {
		logger := &recordingLogger{}
		notifySchemaReady(nil, logger.Printf)
		notifySchemaError(nil, errors.New("boom"), logger.Printf)
		if logger.String() != "" {
			t.Errorf("unexpected log lines:\n%s", logger)
		}
	})

	t.Run("callbacks are called once with their argument", func(t *testing.T) {
		var ready int
		notifySchemaReady(func() { ready++ }, nil)
		if ready != 1 {
			t.Errorf("OnSchemaReady called %d times, want 1", ready)
		}
		want := errors.New("boom")
		var got []error
		notifySchemaError(func(err error) { got = append(got, err) }, want, nil)
		if len(got) != 1 || got[0] != want {
			t.Errorf("OnSchemaError called with %v, want exactly one call with %v", got, want)
		}
	})

	t.Run("a panic is recovered and logged by type only", func(t *testing.T) {
		const secret = "panic-secret-19ab"
		logger := &recordingLogger{}
		notifySchemaReady(func() { panic(secret) }, logger.Printf)
		notifySchemaError(func(error) { panic(fmt.Errorf("wrapped %s: %w", secret, errors.New("cause"))) }, errors.New("boom"), logger.Printf)
		for _, want := range []string{
			"agentanalytics: OnSchemaReady callback failed: recovered panic of type string",
			"agentanalytics: OnSchemaError callback failed: recovered panic of type *fmt.wrapError",
		} {
			if !logger.contains(want) {
				t.Errorf("logger lacks %q; got:\n%s", want, logger)
			}
		}
		if logger.contains(secret) {
			t.Errorf("the log leaks the panic value:\n%s", logger)
		}
	})

	t.Run("a panic is recovered without a logger", func(t *testing.T) {
		notifySchemaReady(func() { panic("no logger") }, nil)
		notifySchemaError(func(error) { panic("no logger") }, errors.New("boom"), nil)
	})
}

// schemaCallbackRecorder counts the schema lifecycle callbacks of a plugin
// under test and keeps the errors that OnSchemaError received.
type schemaCallbackRecorder struct {
	ready atomic.Int32
	mu    sync.Mutex
	errs  []error
	// readyCh receives one value per OnSchemaReady call.
	readyCh chan struct{}
}

func newSchemaCallbackRecorder() *schemaCallbackRecorder {
	return &schemaCallbackRecorder{readyCh: make(chan struct{}, 16)}
}

func (r *schemaCallbackRecorder) install(c *Config) {
	c.OnSchemaReady = func() {
		r.ready.Add(1)
		r.readyCh <- struct{}{}
	}
	c.OnSchemaError = func(err error) {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.errs = append(r.errs, err)
	}
}

func (r *schemaCallbackRecorder) errors() []error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]error(nil), r.errs...)
}

// waitReady waits up to d for OnSchemaReady to have been called n times in
// all and reports whether it was.
func (r *schemaCallbackRecorder) waitReady(n int32, d time.Duration) bool {
	deadline := time.Now().Add(d)
	for r.ready.Load() < n && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	return r.ready.Load() == n
}

// tableTransport answers the BigQuery table API of the events table. Each
// entry of getStatuses is the status of one table metadata request in turn
// (http.StatusOK returns an up-to-date table unless legacy is set, in which
// case the table lacks event_id); postStatus answers table creation;
// patchStatus answers the schema upgrade; view DDL jobs are answered as done
// unless jobHook, if set, returns a response or an error for one.
type tableTransport struct {
	t           *testing.T
	getStatuses []int
	legacy      bool
	postStatus  int
	patchStatus int
	jobHook     func(r *http.Request) (*http.Response, error)

	gets, posts, patches, jobs atomic.Int32
}

func (tr *tableTransport) respond(status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader(body))}
}

func (tr *tableTransport) apiError(status int) *http.Response {
	return tr.respond(status, fmt.Sprintf(`{"error":{"code":%d,"message":"status %d"}}`, status, status))
}

func (tr *tableTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	const tablePath = "/datasets/test_dataset/tables/test-table"
	switch {
	case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, tablePath):
		n := int(tr.gets.Add(1))
		status := http.StatusOK
		if n <= len(tr.getStatuses) {
			status = tr.getStatuses[n-1]
		} else if len(tr.getStatuses) > 0 {
			status = tr.getStatuses[len(tr.getStatuses)-1]
		}
		if status != http.StatusOK {
			return tr.apiError(status), nil
		}
		schema := EventsSchema()
		if tr.legacy {
			var legacy bq.Schema
			for _, f := range schema {
				if f.Name != eventIDField {
					legacy = append(legacy, f)
				}
			}
			schema = legacy
		}
		fields, err := schema.ToJSONFields()
		if err != nil {
			tr.t.Errorf("ToJSONFields: %v", err)
		}
		return tr.respond(http.StatusOK, fmt.Sprintf(`{"etag":"etag-1","schema":{"fields":%s}}`, fields)), nil
	case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/datasets/test_dataset/tables"):
		tr.posts.Add(1)
		if tr.postStatus != 0 && tr.postStatus != http.StatusOK {
			return tr.apiError(tr.postStatus), nil
		}
		body, _ := io.ReadAll(r.Body)
		return tr.respond(http.StatusOK, string(body)), nil
	case (r.Method == http.MethodPatch || r.Method == http.MethodPut) && strings.HasSuffix(r.URL.Path, tablePath):
		tr.patches.Add(1)
		if tr.patchStatus != 0 && tr.patchStatus != http.StatusOK {
			return tr.apiError(tr.patchStatus), nil
		}
		body, _ := io.ReadAll(r.Body)
		return tr.respond(http.StatusOK, string(body)), nil
	case r.Method == http.MethodPost && strings.Contains(r.URL.Path, "/jobs"):
		tr.jobs.Add(1)
		if tr.jobHook != nil {
			if resp, err := tr.jobHook(r); resp != nil || err != nil {
				return resp, err
			}
		}
		return tr.respond(http.StatusOK, `{"jobReference":{"projectId":"test-project","jobId":"test-job"},"status":{"state":"DONE"}}`), nil
	}
	// View metadata reads and anything else: an empty object.
	return tr.respond(http.StatusOK, "{}"), nil
}

// newSchemaCallbackPlugin constructs a plugin whose BigQuery table API is
// transport, with the recorder's callbacks installed, and returns the plugin
// and the construction error.
func newSchemaCallbackPlugin(t *testing.T, transport http.RoundTripper, rec *schemaCallbackRecorder, mutate func(*Config)) (*baseplugin.Plugin, error) {
	t.Helper()
	ctx := context.Background()
	bqClient, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(&http.Client{Transport: transport}))
	if err != nil {
		t.Fatalf("bq.NewClient: %v", err)
	}
	t.Cleanup(func() { _ = bqClient.Close() })
	_, writeClient := newTestWriteClient(t)

	cfg := DefaultConfig()
	cfg.ProjectID = "test-project"
	cfg.DatasetID = "test_dataset"
	cfg.TableName = "test-table"
	rec.install(&cfg)
	if mutate != nil {
		mutate(&cfg)
	}
	p, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, writeClient)
	if p != nil {
		t.Cleanup(func() { _ = p.Close() })
	}
	return p, err
}

func TestPlugin_OnSchemaReady_TablePass(t *testing.T) {
	for _, tc := range []struct {
		name        string
		transport   *tableTransport
		wantPosts   int32
		wantPatches int32
	}{
		{
			name:      "an up-to-date table is ready as read",
			transport: &tableTransport{},
		},
		{
			name:      "a missing table is ready once created",
			transport: &tableTransport{getStatuses: []int{http.StatusNotFound}},
			wantPosts: 1,
		},
		{
			name:        "a legacy table is ready once upgraded",
			transport:   &tableTransport{legacy: true},
			wantPatches: 1,
		},
		{
			name:      "a table created by another process is ready once read again",
			transport: &tableTransport{getStatuses: []int{http.StatusNotFound, http.StatusOK}, postStatus: http.StatusConflict},
			wantPosts: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.transport.t = t
			rec := newSchemaCallbackRecorder()
			logger := &recordingLogger{}
			p, err := newSchemaCallbackPlugin(t, tc.transport, rec, func(c *Config) {
				c.DisableCreateViews = true
				c.Logger = logger
			})
			if err != nil {
				t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
			}
			// With no view pass, OnSchemaReady runs before the constructor returns.
			if got := rec.ready.Load(); got != 1 {
				t.Errorf("OnSchemaReady called %d times when the constructor returned, want 1", got)
			}
			if errs := rec.errors(); len(errs) != 0 {
				t.Errorf("OnSchemaError called with %v, want no calls", errs)
			}
			if got := tc.transport.posts.Load(); got != tc.wantPosts {
				t.Errorf("table creations = %d, want %d", got, tc.wantPosts)
			}
			if got := tc.transport.patches.Load(); got != tc.wantPatches {
				t.Errorf("schema upgrades = %d, want %d", got, tc.wantPatches)
			}
			if got := tc.transport.jobs.Load(); got != 0 {
				t.Errorf("view DDL jobs = %d, want 0 with DisableCreateViews", got)
			}
			if err := p.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}
			if got := rec.ready.Load(); got != 1 {
				t.Errorf("OnSchemaReady called %d times after Close, want 1", got)
			}
			if logger.contains("OnSchema") {
				t.Errorf("unexpected callback log lines:\n%s", logger)
			}
		})
	}
}

func TestPlugin_OnSchemaError(t *testing.T) {
	for _, tc := range []struct {
		name      string
		transport *tableTransport
		wantErr   string
	}{
		{
			name:      "a table that cannot be read",
			transport: &tableTransport{getStatuses: []int{http.StatusForbidden}},
			wantErr:   "failed to get BigQuery table test-table metadata",
		},
		{
			name:      "a table that cannot be created",
			transport: &tableTransport{getStatuses: []int{http.StatusNotFound}, postStatus: http.StatusForbidden},
			wantErr:   "failed to create BigQuery table test-table",
		},
		{
			name:      "a table that cannot be upgraded",
			transport: &tableTransport{legacy: true, patchStatus: http.StatusForbidden},
			wantErr:   "failed to upgrade BigQuery table test-table schema",
		},
		{
			name:      "a table that cannot be read again after a concurrent creation",
			transport: &tableTransport{getStatuses: []int{http.StatusNotFound, http.StatusForbidden}, postStatus: http.StatusConflict},
			wantErr:   "failed to get BigQuery table test-table metadata",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tc.transport.t = t
			rec := newSchemaCallbackRecorder()
			p, err := newSchemaCallbackPlugin(t, tc.transport, rec, nil)
			if p != nil || err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() = %v, %v; want nil and an error containing %q", p, err, tc.wantErr)
			}
			errs := rec.errors()
			if len(errs) != 1 {
				t.Fatalf("OnSchemaError called with %v, want exactly one call", errs)
			}
			// The callback receives the error that the constructor returns.
			if errs[0] != err {
				t.Errorf("OnSchemaError error = %v, want the constructor's error %v", errs[0], err)
			}
			var apiErr *googleapi.Error
			if !errors.As(errs[0], &apiErr) || apiErr.Code != http.StatusForbidden {
				t.Errorf("OnSchemaError error = %v, want it to wrap a BigQuery API error with code 403", errs[0])
			}
			if got := rec.ready.Load(); got != 0 {
				t.Errorf("OnSchemaReady called %d times, want 0", got)
			}
			if got := tc.transport.jobs.Load(); got != 0 {
				t.Errorf("view DDL jobs = %d, want 0 after a table failure", got)
			}
		})
	}

	t.Run("a panic in OnSchemaError is logged and the constructor still fails", func(t *testing.T) {
		logger := &recordingLogger{}
		rec := newSchemaCallbackRecorder()
		var calls atomic.Int32
		p, err := newSchemaCallbackPlugin(t, &tableTransport{t: t, getStatuses: []int{http.StatusForbidden}}, rec, func(c *Config) {
			c.Logger = logger
			c.OnSchemaError = func(error) {
				calls.Add(1)
				panic("schema error handler bug")
			}
		})
		if p != nil || err == nil || !strings.Contains(err.Error(), "failed to get BigQuery table test-table metadata") {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() = %v, %v; want nil and the table error", p, err)
		}
		if got := calls.Load(); got != 1 {
			t.Errorf("OnSchemaError called %d times, want 1", got)
		}
		if want := "agentanalytics: OnSchemaError callback failed: recovered panic of type string"; !logger.contains(want) {
			t.Errorf("logger lacks %q; got:\n%s", want, logger)
		}
	})

	t.Run("a panic in OnSchemaReady is logged and the constructor still succeeds", func(t *testing.T) {
		logger := &recordingLogger{}
		rec := newSchemaCallbackRecorder()
		var calls atomic.Int32
		p, err := newSchemaCallbackPlugin(t, &tableTransport{t: t}, rec, func(c *Config) {
			c.DisableCreateViews = true
			c.Logger = logger
			c.OnSchemaReady = func() {
				calls.Add(1)
				panic(errors.New("schema ready handler bug"))
			}
		})
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		if got := calls.Load(); got != 1 {
			t.Errorf("OnSchemaReady called %d times, want 1", got)
		}
		if want := "agentanalytics: OnSchemaReady callback failed: recovered panic of type *errors.errorString"; !logger.contains(want) {
			t.Errorf("logger lacks %q; got:\n%s", want, logger)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})
}

func TestPlugin_OnSchemaReady_ViewPass(t *testing.T) {
	t.Run("called once from the background pass once every view is created", func(t *testing.T) {
		transport := &tableTransport{t: t}
		rec := newSchemaCallbackRecorder()
		logger := &recordingLogger{}
		p, err := newSchemaCallbackPlugin(t, transport, rec, func(c *Config) { c.Logger = logger })
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		if !rec.waitReady(1, 5*time.Second) {
			t.Fatalf("OnSchemaReady called %d times within 5s, want 1", rec.ready.Load())
		}
		// Every view was created before the callback ran.
		if got, want := transport.jobs.Load(), int32(len(eventViewSpecs)); got != want {
			t.Errorf("view DDL jobs = %d when OnSchemaReady ran, want %d", got, want)
		}
		if errs := rec.errors(); len(errs) != 0 {
			t.Errorf("OnSchemaError called with %v, want no calls", errs)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if got := rec.ready.Load(); got != 1 {
			t.Errorf("OnSchemaReady called %d times after Close, want 1", got)
		}
		if logger.contains("OnSchema") {
			t.Errorf("unexpected callback log lines:\n%s", logger)
		}
	})

	t.Run("not called when a view fails, and OnSchemaError is not called either", func(t *testing.T) {
		transport := &tableTransport{t: t}
		// The second view DDL job is denied; the others succeed.
		var jobIdx atomic.Int32
		transport.jobHook = func(*http.Request) (*http.Response, error) {
			if jobIdx.Add(1) == 2 {
				return transport.apiError(http.StatusForbidden), nil
			}
			return nil, nil
		}
		rec := newSchemaCallbackRecorder()
		logger := &recordingLogger{}
		p, err := newSchemaCallbackPlugin(t, transport, rec, func(c *Config) { c.Logger = logger })
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		// Wait for the pass to log its summary, the last thing it does before
		// it would call OnSchemaReady, then close.
		deadline := time.Now().Add(5 * time.Second)
		for !logger.contains("view creation finished") && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		if want := fmt.Sprintf("view creation finished: %d created, 1 failed of %d; call SyncViews to retry", len(eventViewSpecs)-1, len(eventViewSpecs)); !logger.contains(want) {
			t.Fatalf("logger lacks %q; got:\n%s", want, logger)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if got := rec.ready.Load(); got != 0 {
			t.Errorf("OnSchemaReady called %d times after a view failed, want 0", got)
		}
		if errs := rec.errors(); len(errs) != 0 {
			t.Errorf("OnSchemaError called with %v for a view failure, want no calls", errs)
		}
	})

	t.Run("not called when the view definitions are invalid", func(t *testing.T) {
		transport := &tableTransport{t: t}
		rec := newSchemaCallbackRecorder()
		logger := &recordingLogger{}
		p, err := newSchemaCallbackPlugin(t, transport, rec, func(c *Config) {
			// A view would collide with the events table.
			c.TableName = "v_llm_request"
			c.Logger = logger
		})
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		if !logger.contains("agentanalytics: skipping view creation") {
			t.Fatalf("startup log does not report skipped view creation; got:\n%s", logger)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if got := rec.ready.Load(); got != 0 {
			t.Errorf("OnSchemaReady called %d times with invalid view definitions, want 0", got)
		}
		if errs := rec.errors(); len(errs) != 0 {
			t.Errorf("OnSchemaError called with %v, want no calls", errs)
		}
	})

	t.Run("not called by a startup pass that SyncViews preempts, nor by SyncViews", func(t *testing.T) {
		started := make(chan struct{})
		var startOnce sync.Once
		var blockedOne atomic.Bool
		transport := &tableTransport{t: t}
		transport.jobHook = func(r *http.Request) (*http.Response, error) {
			if blockedOne.CompareAndSwap(false, true) {
				startOnce.Do(func() { close(started) })
				// The first job of the startup pass waits until the pass is
				// canceled, as a slow BigQuery job would.
				<-r.Context().Done()
				return nil, r.Context().Err()
			}
			return nil, nil
		}
		rec := newSchemaCallbackRecorder()
		p, err := newSchemaCallbackPlugin(t, transport, rec, nil)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		select {
		case <-started:
		case <-time.After(5 * time.Second):
			t.Fatal("the startup view pass did not start a job within 5s")
		}
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		if err := bqaap.SyncViews(context.Background()); err != nil {
			t.Fatalf("SyncViews() error = %v", err)
		}
		// SyncViews ran the whole pass of its own after the startup pass ended.
		if got, want := transport.jobs.Load(), int32(len(eventViewSpecs)+1); got != want {
			t.Errorf("view DDL jobs = %d after SyncViews, want %d", got, want)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if got := rec.ready.Load(); got != 0 {
			t.Errorf("OnSchemaReady called %d times, want 0: the startup pass was preempted and SyncViews never calls it", got)
		}
	})

	t.Run("not called when Close ends an in-flight startup pass on the final view", func(t *testing.T) {
		lastViewStarted := make(chan struct{})
		var lastOnce sync.Once
		transport := &tableTransport{t: t}
		transport.jobHook = func(r *http.Request) (*http.Response, error) {
			if transport.jobs.Load() == int32(len(eventViewSpecs)) {
				lastOnce.Do(func() { close(lastViewStarted) })
				// Wait until Close cancels viewsCtx, then let the final job
				// report DONE cleanly as a concurrent BigQuery job would.
				<-r.Context().Done()
			}
			return nil, nil
		}
		rec := newSchemaCallbackRecorder()
		p, err := newSchemaCallbackPlugin(t, transport, rec, nil)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		select {
		case <-lastViewStarted:
		case <-time.After(5 * time.Second):
			t.Fatal("the startup view pass did not reach the final view within 5s")
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if got := rec.ready.Load(); got != 0 {
			t.Errorf("OnSchemaReady called %d times after Close preempted the final view, want 0", got)
		}
	})

	t.Run("not called when SyncViews preempts an in-flight startup pass on the final view", func(t *testing.T) {
		lastViewStarted := make(chan struct{})
		var lastOnce sync.Once
		transport := &tableTransport{t: t}
		transport.jobHook = func(r *http.Request) (*http.Response, error) {
			if transport.jobs.Load() == int32(len(eventViewSpecs)) {
				lastOnce.Do(func() {
					close(lastViewStarted)
					// Block only the final job of the startup pass until
					// SyncViews cancels viewsCtx, then report DONE cleanly.
					<-r.Context().Done()
				})
			}
			return nil, nil
		}
		rec := newSchemaCallbackRecorder()
		p, err := newSchemaCallbackPlugin(t, transport, rec, nil)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		select {
		case <-lastViewStarted:
		case <-time.After(5 * time.Second):
			t.Fatal("the startup view pass did not reach the final view within 5s")
		}
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		if err := bqaap.SyncViews(context.Background()); err != nil {
			t.Fatalf("SyncViews() error = %v", err)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if got := rec.ready.Load(); got != 0 {
			t.Errorf("OnSchemaReady called %d times after SyncViews preempted the final view, want 0", got)
		}
	})

	t.Run("SyncViews after a ready schema does not call it again", func(t *testing.T) {
		transport := &tableTransport{t: t}
		rec := newSchemaCallbackRecorder()
		p, err := newSchemaCallbackPlugin(t, transport, rec, nil)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		if !rec.waitReady(1, 5*time.Second) {
			t.Fatalf("OnSchemaReady called %d times within 5s, want 1", rec.ready.Load())
		}
		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		if err := bqaap.SyncViews(context.Background()); err != nil {
			t.Fatalf("SyncViews() error = %v", err)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if got := rec.ready.Load(); got != 1 {
			t.Errorf("OnSchemaReady called %d times, want 1", got)
		}
	})

	t.Run("a panic in OnSchemaReady from the background pass is logged through a non-default Logger", func(t *testing.T) {
		logger := &recordingLogger{}
		rec := newSchemaCallbackRecorder()
		var calls atomic.Int32
		p, err := newSchemaCallbackPlugin(t, &tableTransport{t: t}, rec, func(c *Config) {
			c.Logger = logger
			c.OnSchemaReady = func() {
				calls.Add(1)
				panic("schema ready handler bug")
			}
		})
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		want := "agentanalytics: OnSchemaReady callback failed: recovered panic of type string"
		deadline := time.Now().Add(5 * time.Second)
		for !logger.contains(want) && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		if !logger.contains(want) {
			t.Errorf("logger lacks %q; got:\n%s", want, logger)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if got := calls.Load(); got != 1 {
			t.Errorf("OnSchemaReady called %d times, want 1", got)
		}
	})
}

func TestPlugin_SchemaCallbacks_NoPass(t *testing.T) {
	t.Run("a plugin built without a BigQuery client calls neither callback", func(t *testing.T) {
		rec := newSchemaCallbackRecorder()
		_, writeClient := newTestWriteClient(t)
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "test-table"
		rec.install(&cfg)
		p, err := NewBigQueryAgentAnalyticsPluginWithClients(context.Background(), cfg, nil, writeClient)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if got := rec.ready.Load(); got != 0 {
			t.Errorf("OnSchemaReady called %d times without a BigQuery client, want 0", got)
		}
		if errs := rec.errors(); len(errs) != 0 {
			t.Errorf("OnSchemaError called with %v, want no calls", errs)
		}
	})

	t.Run("a disabled plugin calls neither callback", func(t *testing.T) {
		rec := newSchemaCallbackRecorder()
		transport := &tableTransport{t: t, getStatuses: []int{http.StatusForbidden}}
		p, err := newSchemaCallbackPlugin(t, transport, rec, func(c *Config) { c.Enabled = false })
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if got := transport.gets.Load(); got != 0 {
			t.Errorf("table metadata requests = %d, want 0 for a disabled plugin", got)
		}
		if got := rec.ready.Load(); got != 0 {
			t.Errorf("OnSchemaReady called %d times for a disabled plugin, want 0", got)
		}
		if errs := rec.errors(); len(errs) != 0 {
			t.Errorf("OnSchemaError called with %v for a disabled plugin, want no calls", errs)
		}
	})

	t.Run("NewBigQueryAgentAnalyticsPluginWithConfig disabled calls neither callback", func(t *testing.T) {
		rec := newSchemaCallbackRecorder()
		cfg := Config{Enabled: false}
		rec.install(&cfg)
		p, err := NewBigQueryAgentAnalyticsPluginWithConfig(context.Background(), cfg)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithConfig() error = %v", err)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if got := rec.ready.Load(); got != 0 {
			t.Errorf("OnSchemaReady called %d times for a disabled plugin, want 0", got)
		}
		if errs := rec.errors(); len(errs) != 0 {
			t.Errorf("OnSchemaError called with %v for a disabled plugin, want no calls", errs)
		}
	})
}

// TestPlugin_SchemaCallbacks_ConcurrentPlugins constructs several plugins at
// once that share one OnSchemaReady, as a host that builds a plugin per
// request does, and checks under -race that each calls it exactly once.
func TestPlugin_SchemaCallbacks_ConcurrentPlugins(t *testing.T) {
	const plugins = 6
	var ready atomic.Int32
	t.Run("construct", func(t *testing.T) {
		for i := 0; i < plugins; i++ {
			withViews := i%2 != 0
			t.Run(fmt.Sprintf("plugin-%d", i), func(t *testing.T) {
				t.Parallel()
				transport := &tableTransport{t: t}
				rec := newSchemaCallbackRecorder()
				p, err := newSchemaCallbackPlugin(t, transport, rec, func(c *Config) {
					c.DisableCreateViews = !withViews
					record := c.OnSchemaReady
					c.OnSchemaReady = func() {
						record()
						ready.Add(1)
					}
				})
				if err != nil {
					t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
				}
				if !rec.waitReady(1, 5*time.Second) {
					t.Fatalf("OnSchemaReady called %d times within 5s, want 1", rec.ready.Load())
				}
				if err := p.Close(); err != nil {
					t.Fatalf("Close() error = %v", err)
				}
				if got := rec.ready.Load(); got != 1 {
					t.Errorf("OnSchemaReady called %d times after Close, want 1", got)
				}
			})
		}
	})
	if got := ready.Load(); got != plugins {
		t.Errorf("OnSchemaReady called %d times across %d plugins, want %d", got, plugins, plugins)
	}
}
