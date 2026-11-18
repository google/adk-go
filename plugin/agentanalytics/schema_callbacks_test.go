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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"weak"

	bq "cloud.google.com/go/bigquery"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/retryandreflect"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
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

	t.Run("a panicking logger in recovery does not escape", func(t *testing.T) {
		panicLogf := func(string, ...any) { panic("logger boom") }
		notifySchemaReady(func() { panic("ready boom") }, panicLogf)
		notifySchemaError(func(error) { panic("error boom") }, errors.New("boom"), panicLogf)
	})
}

// schemaCallbackRecorder counts the schema lifecycle callbacks of a plugin
// under test and keeps the errors that OnSchemaError received.
type schemaCallbackRecorder struct {
	ready       atomic.Int32
	mu          sync.Mutex
	readyNotify chan struct{}
	errs        []error
}

func newSchemaCallbackRecorder() *schemaCallbackRecorder {
	return &schemaCallbackRecorder{}
}

func (r *schemaCallbackRecorder) install(c *Config) {
	c.OnSchemaReady = func() {
		r.mu.Lock()
		r.ready.Add(1)
		if r.readyNotify != nil {
			close(r.readyNotify)
			r.readyNotify = nil
		}
		r.mu.Unlock()
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
	timer := time.NewTimer(d)
	defer timer.Stop()
	for {
		r.mu.Lock()
		if r.ready.Load() >= n {
			r.mu.Unlock()
			return r.ready.Load() == n
		}
		if r.readyNotify == nil {
			r.readyNotify = make(chan struct{})
		}
		ch := r.readyNotify
		r.mu.Unlock()

		select {
		case <-ch:
		case <-timer.C:
			return r.ready.Load() == n
		}
	}
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
	viewHook    func(r *http.Request) (*http.Response, error)

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
		return tr.respond(http.StatusOK, doneJobResponse), nil
	}
	if tr.viewHook != nil {
		if resp, err := tr.viewHook(r); resp != nil || err != nil {
			return resp, err
		}
	}
	// View metadata reads and anything else: an empty object.
	return tr.respond(http.StatusOK, "{}"), nil
}

// doneJobResponse is the jobs.insert response of a DDL job that is done.
const doneJobResponse = `{"jobReference":{"projectId":"test-project","jobId":"test-job"},"status":{"state":"DONE"}}`

// finalViewJobResponse returns the response of a view DDL job request r whose
// job is done. When the client closes the response body, after it decoded the
// response, the body calls end and returns once r's context is done, so that
// end can cancel the view pass that sent r after the pass's job succeeded.
func finalViewJobResponse(r *http.Request, end func()) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Body: &hookedCloseBody{
			Reader: strings.NewReader(doneJobResponse),
			onClose: func() {
				end()
				<-r.Context().Done()
			},
		},
	}
}

// hookedCloseBody is a response body that runs onClose when it is first
// closed.
type hookedCloseBody struct {
	io.Reader
	once    sync.Once
	onClose func()
}

func (b *hookedCloseBody) Close() error {
	b.once.Do(b.onClose)
	return nil
}

// waitAsyncViewPass waits for an in-flight startup view pass, if one is still
// running, to finish cleanly.
func waitAsyncViewPass(t *testing.T, p *baseplugin.Plugin) {
	t.Helper()
	bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
	if !ok {
		t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
	}
	bqaap.pluginState.viewsMu.Lock()
	done := bqaap.pluginState.asyncViewsDone
	bqaap.pluginState.viewsMu.Unlock()
	if done != nil {
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("startup view pass did not finish within 5s")
		}
	}
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
		logger.waitContains("view creation finished", 5*time.Second)
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

	// The two subtests below end the startup pass after its final view DDL
	// job succeeded: the BigQuery client has decoded the job's DONE response
	// when it closes the response body, and the body's Close starts Close or
	// SyncViews and returns once they canceled the pass, which the request's
	// context derives from. The pass then sees every view created, and only
	// its check that it was not ended keeps it from calling OnSchemaReady.
	t.Run("not called when Close ends an in-flight startup pass on the final view", func(t *testing.T) {
		transport := &tableTransport{t: t}
		pluginReady := make(chan struct{})
		var p *baseplugin.Plugin
		closeErr := make(chan error, 1)
		transport.jobHook = func(r *http.Request) (*http.Response, error) {
			if transport.jobs.Load() != int32(len(eventViewSpecs)) {
				return nil, nil
			}
			return finalViewJobResponse(r, func() {
				<-pluginReady
				go func() { closeErr <- p.Close() }()
			}), nil
		}
		rec := newSchemaCallbackRecorder()
		var err error
		p, err = newSchemaCallbackPlugin(t, transport, rec, nil)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		close(pluginReady)
		select {
		case err := <-closeErr:
			if err != nil {
				t.Fatalf("Close() error = %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("Close did not end the startup view pass on its final view within 5s")
		}
		if got := rec.ready.Load(); got != 0 {
			t.Errorf("OnSchemaReady called %d times after Close ended the pass on its final view, want 0", got)
		}
	})

	t.Run("not called when SyncViews preempts an in-flight startup pass on the final view", func(t *testing.T) {
		transport := &tableTransport{t: t}
		pluginReady := make(chan struct{})
		var bqaap *BigQueryAgentAnalyticsPlugin
		syncErr := make(chan error, 1)
		transport.jobHook = func(r *http.Request) (*http.Response, error) {
			// Only the final job of the startup pass: SyncViews's own jobs
			// come after it.
			if transport.jobs.Load() != int32(len(eventViewSpecs)) {
				return nil, nil
			}
			return finalViewJobResponse(r, func() {
				<-pluginReady
				go func() { syncErr <- bqaap.SyncViews(context.Background()) }()
			}), nil
		}
		rec := newSchemaCallbackRecorder()
		p, err := newSchemaCallbackPlugin(t, transport, rec, nil)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		var ok bool
		if bqaap, ok = AsBigQueryAgentAnalyticsPlugin(p); !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		close(pluginReady)
		select {
		case err := <-syncErr:
			if err != nil {
				t.Fatalf("SyncViews() error = %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("SyncViews did not preempt the startup view pass on its final view within 5s")
		}
		// SyncViews ran the whole pass of its own after the startup pass ended.
		if got, want := transport.jobs.Load(), int32(2*len(eventViewSpecs)); got != want {
			t.Errorf("view DDL jobs = %d after SyncViews, want %d", got, want)
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
		if !logger.waitContains(want, 5*time.Second) {
			t.Errorf("logger lacks %q; got:\n%s", want, logger)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if got := calls.Load(); got != 1 {
			t.Errorf("OnSchemaReady called %d times, want 1", got)
		}
	})

	t.Run("not called when a retried DDL job gets 409 while the target view returns 404", func(t *testing.T) {
		transport := &tableTransport{t: t}
		var (
			mu          sync.Mutex
			firstJobID  string
			sameIDRetry atomic.Bool
		)
		transport.viewHook = func(r *http.Request) (*http.Response, error) {
			if r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/datasets/test_dataset/tables/v_llm_request") {
				return transport.apiError(http.StatusNotFound), nil
			}
			return nil, nil
		}
		transport.jobHook = func(r *http.Request) (*http.Response, error) {
			body, err := io.ReadAll(r.Body)
			if err != nil {
				return nil, err
			}
			var payload struct {
				JobReference struct {
					JobID string `json:"jobId"`
				} `json:"jobReference"`
				Configuration struct {
					Query struct {
						Query string `json:"query"`
					} `json:"query"`
				} `json:"configuration"`
			}
			if err := json.Unmarshal(body, &payload); err != nil {
				return nil, err
			}
			if !strings.Contains(payload.Configuration.Query.Query, "v_llm_request") {
				return nil, nil
			}
			mu.Lock()
			defer mu.Unlock()
			if firstJobID == "" {
				firstJobID = payload.JobReference.JobID
				return transport.apiError(http.StatusServiceUnavailable), nil
			}
			if payload.JobReference.JobID == firstJobID {
				sameIDRetry.Store(true)
			}
			return transport.apiError(http.StatusConflict), nil
		}

		rec := newSchemaCallbackRecorder()
		logger := &recordingLogger{}
		p, err := newSchemaCallbackPlugin(t, transport, rec, func(c *Config) { c.Logger = logger })
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		waitAsyncViewPass(t, p)
		if !sameIDRetry.Load() {
			t.Fatal("expected jobs.insert retry to reuse the same jobId and hit 409")
		}
		if got := rec.ready.Load(); got != 0 {
			t.Errorf("OnSchemaReady called %d times while v_llm_request returned 404, want 0", got)
		}
		if want := fmt.Sprintf("view creation finished: %d created, 1 failed of %d; call SyncViews to retry", len(eventViewSpecs)-1, len(eventViewSpecs)); !logger.contains(want) {
			t.Errorf("logger lacks %q; got:\n%s", want, logger)
		}

		bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
		if !ok {
			t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
		}
		if err := bqaap.SyncViews(context.Background()); err == nil {
			t.Error("SyncViews() = nil while v_llm_request returns 404 after job 409, want error")
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})

	t.Run("not called when a retried DDL job gets 409 while an existing view still has a stale query", func(t *testing.T) {
		transport := &tableTransport{t: t}
		var (
			mu         sync.Mutex
			firstJobID string
		)
		transport.viewHook = func(r *http.Request) (*http.Response, error) {
			if r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/datasets/test_dataset/tables/v_llm_request") {
				return transport.respond(http.StatusOK, `{"type":"VIEW","view":{"query":"SELECT 1"}}`), nil
			}
			return nil, nil
		}
		transport.jobHook = func(r *http.Request) (*http.Response, error) {
			body, err := io.ReadAll(r.Body)
			if err != nil {
				return nil, err
			}
			if !strings.Contains(string(body), "v_llm_request") {
				return nil, nil
			}
			mu.Lock()
			defer mu.Unlock()
			if firstJobID == "" {
				firstJobID = "seen"
				return transport.apiError(http.StatusServiceUnavailable), nil
			}
			return transport.apiError(http.StatusConflict), nil
		}

		rec := newSchemaCallbackRecorder()
		p, err := newSchemaCallbackPlugin(t, transport, rec, nil)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		waitAsyncViewPass(t, p)
		if got := rec.ready.Load(); got != 0 {
			t.Errorf("OnSchemaReady called %d times while v_llm_request retained a stale query, want 0", got)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})

	t.Run("called when a retried DDL job gets 409 after the first attempt created the view", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.ProjectID = "test-project"
		cfg.DatasetID = "test_dataset"
		cfg.TableName = "test-table"
		defs, err := cfg.ViewDefinitions()
		if err != nil {
			t.Fatalf("ViewDefinitions() error = %v", err)
		}
		var wantSelectBody string
		for _, def := range defs {
			if def.Name == "v_llm_request" {
				_, wantSelectBody, _ = strings.Cut(def.Query, " AS\n")
				break
			}
		}
		viewJSON, err := json.Marshal(map[string]any{
			"type": "VIEW",
			"view": map[string]string{"query": wantSelectBody},
		})
		if err != nil {
			t.Fatalf("json.Marshal: %v", err)
		}

		transport := &tableTransport{t: t}
		var viewCreated atomic.Bool
		transport.viewHook = func(r *http.Request) (*http.Response, error) {
			if r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/datasets/test_dataset/tables/v_llm_request") {
				if viewCreated.Load() {
					return transport.respond(http.StatusOK, string(viewJSON)), nil
				}
				return transport.apiError(http.StatusNotFound), nil
			}
			return nil, nil
		}
		transport.jobHook = func(r *http.Request) (*http.Response, error) {
			body, err := io.ReadAll(r.Body)
			if err != nil {
				return nil, err
			}
			if !strings.Contains(string(body), "v_llm_request") {
				return nil, nil
			}
			if viewCreated.CompareAndSwap(false, true) {
				return transport.apiError(http.StatusServiceUnavailable), nil
			}
			return transport.apiError(http.StatusConflict), nil
		}

		rec := newSchemaCallbackRecorder()
		p, err := newSchemaCallbackPlugin(t, transport, rec, nil)
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		waitAsyncViewPass(t, p)
		if got := rec.ready.Load(); got != 1 {
			t.Errorf("OnSchemaReady called %d times after view was confirmed in place, want 1", got)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
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
		for i := range plugins {
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

type panickingPrintfLogger struct{}

func (panickingPrintfLogger) Printf(string, ...any) {
	panic("logger Printf boom")
}

func (panickingPrintfLogger) Println(...any) {
	panic("logger Println boom")
}

func TestPlugin_PanickingLoggerInRecovery(t *testing.T) {
	t.Run("OnSchemaError + panicking Logger returns table error cleanly", func(t *testing.T) {
		transport := &tableTransport{t: t, getStatuses: []int{http.StatusForbidden}}
		rec := newSchemaCallbackRecorder()
		var apiErr *googleapi.Error
		p, err := newSchemaCallbackPlugin(t, transport, rec, func(c *Config) {
			c.Logger = panickingPrintfLogger{}
			c.OnSchemaError = func(error) {
				panic("onSchemaError boom")
			}
		})
		if p != nil {
			_ = p.Close()
			t.Fatal("expected nil plugin when table check fails")
		}
		if !errors.As(err, &apiErr) || apiErr.Code != http.StatusForbidden {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v, want HTTP 403", err)
		}
	})

	t.Run("OnSchemaReady + panicking Logger on background view pass does not crash", func(t *testing.T) {
		rec := newSchemaCallbackRecorder()
		var calls atomic.Int32
		p, err := newSchemaCallbackPlugin(t, &tableTransport{t: t}, rec, func(c *Config) {
			c.Logger = panickingPrintfLogger{}
			c.OnSchemaReady = func() {
				calls.Add(1)
				panic("onSchemaReady boom")
			}
		})
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		waitAsyncViewPass(t, p)
		if got := calls.Load(); got != 1 {
			t.Errorf("OnSchemaReady called %d times, want 1", got)
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})

	t.Run("ToolResultClassifier + panicking Logger isolates secondary panic", func(t *testing.T) {
		for _, tc := range []struct {
			name       string
			classifier ToolResultClassifier
		}{
			{
				name: "classifier panics",
				classifier: func(agent.Context, tool.Tool, map[string]any, map[string]any) ToolResultVerdict {
					panic("classifier boom")
				},
			},
			{
				name: "classifier returns an undefined decision",
				classifier: func(agent.Context, tool.Tool, map[string]any, map[string]any) ToolResultVerdict {
					return ToolResultVerdict{Decision: ToolResultDecision(99)}
				},
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var calls atomic.Int32
				p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
					c.Logger = panickingPrintfLogger{}
					c.ToolResultClassifier = func(ctx agent.Context, tl tool.Tool, args, result map[string]any) ToolResultVerdict {
						calls.Add(1)
						return tc.classifier(ctx, tl, args, result)
					}
				})
				runLookupCall(t, map[string]any{"status": "ok"}, nil, p)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				if got := calls.Load(); got != 1 {
					t.Errorf("classifier called %d times, want 1", got)
				}
				// The warning that the logger panics on is isolated inside the
				// classifier's recovery, so the call keeps its TOOL_COMPLETED
				// row rather than having AfterToolCallback's recovery drop it.
				if got := len(rowsOfType(rows, EventToolCompleted)); got != 1 {
					t.Errorf("got %d TOOL_COMPLETED rows, want 1; events: %v", got, eventTypes(rows))
				}
				if got := len(rowsOfType(rows, EventToolError)); got != 0 {
					t.Errorf("got %d TOOL_ERROR rows, want 0; events: %v", got, eventTypes(rows))
				}
				requireDropStats(t, p, DropStats{})
			})
		}
	})

	t.Run("recoverCallback + panicking Logger isolates secondary panic", func(t *testing.T) {
		// A ContentFormatter that fails makes the plugin warn through the
		// Logger while it builds the TOOL_COMPLETED row of AfterToolCallback.
		// The Logger panics there, and again when the callback's recovery
		// reports the dropped row: the second panic must not reach the agent.
		p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
			c.Logger = panickingPrintfLogger{}
			c.ContentFormatter = func(payload any, eventType string) (any, error) {
				if eventType == EventToolCompleted {
					return nil, errors.New("formatter failed")
				}
				return payload, nil
			}
		})
		runLookupCall(t, map[string]any{"status": "ok"}, nil, p)

		rows := collectRows(t, requests, untilInvocationEnd)
		requireInvocationCompletedOK(t, rows)
		if got := len(rowsOfType(rows, EventToolStarting)); got != 1 {
			t.Errorf("got %d TOOL_STARTING rows, want 1; events: %v", got, eventTypes(rows))
		}
		if got := len(rowsOfType(rows, EventToolCompleted)); got != 0 {
			t.Errorf("got %d TOOL_COMPLETED rows, want 0: the recovered row is dropped; events: %v", got, eventTypes(rows))
		}
		requireDropStats(t, p, DropStats{AppendFailureDrops: 1, TotalDrops: 1})
	})
}

// requireDropStats checks the DropStats of the analytics plugin p.
func requireDropStats(t *testing.T, p *baseplugin.Plugin, want DropStats) {
	t.Helper()
	bqaap, ok := AsBigQueryAgentAnalyticsPlugin(p)
	if !ok {
		t.Fatal("AsBigQueryAgentAnalyticsPlugin(p) = false")
	}
	if diff := cmp.Diff(want, bqaap.DropStats()); diff != "" {
		t.Errorf("DropStats() mismatch (-want +got):\n%s", diff)
	}
}

// runLookupCall runs one call of a function tool named lookup, which returns
// result and toolErr, through a real runner with plugins. The model calls the
// tool and then answers with text.
func runLookupCall(t *testing.T, result map[string]any, toolErr error, plugins ...*baseplugin.Plugin) {
	t.Helper()
	type lookupArgs struct {
		Query string `json:"query"`
	}
	lookup, err := functiontool.New(
		functiontool.Config{Name: "lookup", Description: "Looks up a value"},
		func(agent.Context, lookupArgs) (map[string]any, error) {
			if toolErr != nil {
				return nil, toolErr
			}
			return maps.Clone(result), nil
		},
	)
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}
	bot, err := llmagent.New(llmagent.Config{
		Name:  "bot",
		Tools: []tool.Tool{lookup},
		Model: &scriptedLLM{responses: []*model.LLMResponse{
			{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID:   "call-lookup",
				Name: "lookup",
				Args: map[string]any{"query": "status"},
			}}}}},
			{Content: textContent("model", "done"), TurnComplete: true},
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	for _, runErr := range newTestRunner(t, bot, plugins...).Run(t.Context(), "u1", "s1", textContent("user", "go"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("runner.Run error: %v", runErr)
		}
	}
}

// TestDocParity_ToolResultClassifierBuiltInRules pins the doc.go difference
// from adk-python: the plugin has no built-in rule for plugin/retryandreflect
// answers, while, as in adk-python, its built-in rule records top-level
// isError and is_error results as TOOL_ERROR unless Config.ToolResultClassifier
// decides otherwise. Each result goes through a real runner: a ToolResultError
// verdict records it as TOOL_ERROR and a ToolResultOK verdict as
// TOOL_COMPLETED, and without either the built-in rule decides.
func TestDocParity_ToolResultClassifierBuiltInRules(t *testing.T) {
	// reflectAndRetryAnswer is the response_type of the answer with which
	// plugin/retryandreflect replaces a tool error.
	const reflectAndRetryAnswer = "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"
	envelopes := []struct {
		name string
		// result and toolErr are what the tool returns.
		result  map[string]any
		toolErr error
		// retry registers plugin/retryandreflect before the analytics plugin.
		retry bool
		// wantKey and wantValue are a field of the result that the
		// classifier sees.
		wantKey   string
		wantValue any
		// builtInError is the error_message of the TOOL_ERROR row with which
		// the built-in rule records the result, or "" if it has no rule for
		// it.
		builtInError string
	}{
		{
			name:      "plugin/retryandreflect answer",
			toolErr:   errors.New("invalid query"),
			retry:     true,
			wantKey:   "response_type",
			wantValue: reflectAndRetryAnswer,
		},
		{
			name:         "top-level isError true",
			result:       map[string]any{"isError": true, "content": "failed"},
			wantKey:      "isError",
			wantValue:    true,
			builtInError: mcpToolErrorMessage,
		},
		{
			name:         "top-level is_error true",
			result:       map[string]any{"is_error": true, "content": "failed"},
			wantKey:      "is_error",
			wantValue:    true,
			builtInError: mcpToolErrorMessage,
		},
	}
	classifiers := []struct {
		name          string
		nilClassifier bool
		verdict       ToolResultVerdict
	}{
		{name: "nil classifier", nilClassifier: true},
		{name: "abstaining classifier", verdict: ToolResultVerdict{Decision: ToolResultAbstain}},
		{name: "OK classifier", verdict: ToolResultVerdict{Decision: ToolResultOK}},
		{name: "explicit error classifier", verdict: ToolResultVerdict{Decision: ToolResultError}},
	}

	for _, env := range envelopes {
		for _, cl := range classifiers {
			t.Run(env.name+"/"+cl.name, func(t *testing.T) {
				// wantErrMessage is the error_message of the call's TOOL_ERROR
				// row, or "" when the call is recorded as TOOL_COMPLETED. A
				// nil classifier decides as an abstaining one does.
				var wantErrMessage string
				switch cl.verdict.Decision {
				case ToolResultError:
					wantErrMessage = classifiedToolErrorMessage
				case ToolResultAbstain:
					wantErrMessage = env.builtInError
				}
				wantToolErrors := 0
				if wantErrMessage != "" {
					wantToolErrors = 1
				}
				var (
					calls []classifierCall
					mu    sync.Mutex
				)
				p, requests, _ := setupTestPluginWithConfig(t, func(c *Config) {
					if !cl.nilClassifier {
						c.ToolResultClassifier = recordingClassifier(&calls, &mu, cl.verdict)
					}
				})
				plugins := []*baseplugin.Plugin{p}
				if env.retry {
					retryPlugin, err := retryandreflect.New()
					if err != nil {
						t.Fatalf("retryandreflect.New: %v", err)
					}
					plugins = []*baseplugin.Plugin{retryPlugin, p}
				}
				runLookupCall(t, env.result, env.toolErr, plugins...)

				rows := collectRows(t, requests, untilInvocationEnd)
				requireInvocationCompletedOK(t, rows)
				mu.Lock()
				gotCalls := slices.Clone(calls)
				mu.Unlock()
				wantCalls := 1
				if cl.nilClassifier {
					wantCalls = 0
				}
				if len(gotCalls) != wantCalls {
					t.Fatalf("classifier called %d times, want %d", len(gotCalls), wantCalls)
				}
				for _, call := range gotCalls {
					if got := call.result[env.wantKey]; got != env.wantValue {
						t.Errorf("classifier result[%q] = %v, want %v", env.wantKey, got, env.wantValue)
					}
				}
				toolErrRows := rowsOfType(rows, EventToolError)
				if got := len(toolErrRows); got != wantToolErrors {
					t.Fatalf("got %d TOOL_ERROR rows, want %d; events: %v", got, wantToolErrors, eventTypes(rows))
				}
				for _, row := range toolErrRows {
					if got := row["error_message"]; got != wantErrMessage {
						t.Errorf("TOOL_ERROR error_message = %v, want %q", got, wantErrMessage)
					}
				}
				completedRows := rowsOfType(rows, EventToolCompleted)
				if got, want := len(completedRows), 1-wantToolErrors; got != want {
					t.Fatalf("got %d TOOL_COMPLETED rows, want %d; events: %v", got, want, eventTypes(rows))
				}
				for _, row := range completedRows {
					want, err := json.Marshal(env.wantValue)
					if err != nil {
						t.Fatalf("json.Marshal: %v", err)
					}
					checkJSONAt(t, jsonObjectColumn(t, row, "content"), string(want), "result", env.wantKey)
				}
			})
		}
	}
}

func TestDocParity_SchemaCallbacksAtMostOnce(t *testing.T) {
	t.Run("table failure calls OnSchemaError once and fails construction", func(t *testing.T) {
		var readyCalls, errCalls atomic.Int32
		p, err := newSchemaCallbackPlugin(t, &tableTransport{t: t, getStatuses: []int{http.StatusForbidden}}, newSchemaCallbackRecorder(), func(c *Config) {
			c.OnSchemaReady = func() { readyCalls.Add(1) }
			c.OnSchemaError = func(error) { errCalls.Add(1) }
		})
		if err == nil || p != nil {
			if p != nil {
				_ = p.Close()
			}
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() = (%v, %v), want nil plugin and error", p, err)
		}
		if readyCalls.Load() != 0 || errCalls.Load() != 1 {
			t.Fatalf("callbacks: ready=%d error=%d, want ready=0 error=1", readyCalls.Load(), errCalls.Load())
		}
	})

	t.Run("ready plugin calls OnSchemaReady once", func(t *testing.T) {
		var readyCalls, errCalls atomic.Int32
		p, err := newSchemaCallbackPlugin(t, &tableTransport{t: t}, newSchemaCallbackRecorder(), func(c *Config) {
			c.OnSchemaReady = func() { readyCalls.Add(1) }
			c.OnSchemaError = func(error) { errCalls.Add(1) }
		})
		if err != nil {
			t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
		}
		waitAsyncViewPass(t, p)
		if readyCalls.Load() != 1 || errCalls.Load() != 0 {
			t.Fatalf("callbacks: ready=%d error=%d, want ready=1 error=0", readyCalls.Load(), errCalls.Load())
		}
		if err := p.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})
}

func TestPlugin_SchemaCallbacks_ClosuresReleased(t *testing.T) {
	for _, disableViews := range []bool{true, false} {
		name := "async_views"
		if disableViews {
			name = "views_disabled"
		}
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			transport := &tableTransport{t: t}
			bqClient, err := bq.NewClient(ctx, "test-project", option.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("bq.NewClient: %v", err)
			}
			t.Cleanup(func() { _ = bqClient.Close() })
			_, writeClient := newTestWriteClient(t)

			var (
				weakReady weak.Pointer[[1024]byte]
				weakErr   weak.Pointer[[1024]byte]
				ready     atomic.Int32
			)
			p := func() *baseplugin.Plugin {
				readyCaptured := new([1024]byte)
				errCaptured := new([1024]byte)
				weakReady = weak.Make(readyCaptured)
				weakErr = weak.Make(errCaptured)

				cfg := DefaultConfig()
				cfg.ProjectID = "test-project"
				cfg.DatasetID = "test_dataset"
				cfg.TableName = "test-table"
				cfg.DisableCreateViews = disableViews
				cfg.OnSchemaReady = func() {
					readyCaptured[0] = 1
					ready.Add(1)
				}
				cfg.OnSchemaError = func(error) {
					errCaptured[0] = 1
				}
				plug, err := NewBigQueryAgentAnalyticsPluginWithClients(ctx, cfg, bqClient, writeClient)
				if err != nil {
					t.Fatalf("NewBigQueryAgentAnalyticsPluginWithClients() error = %v", err)
				}
				return plug
			}()
			defer func() {
				if err := p.Close(); err != nil {
					t.Errorf("Close() error = %v", err)
				}
			}()

			waitAsyncViewPass(t, p)
			if got := ready.Load(); got != 1 {
				t.Fatalf("OnSchemaReady called %d times, want 1", got)
			}

			for range 5 {
				runtime.Gosched()
				runtime.GC()
				if weakReady.Value() == nil && weakErr.Value() == nil {
					break
				}
			}
			if weakReady.Value() != nil {
				t.Error("OnSchemaReady closure capture remained reachable on live plugin")
			}
			if weakErr.Value() != nil {
				t.Error("OnSchemaError closure capture remained reachable on live plugin")
			}
			runtime.KeepAlive(p)
		})
	}
}
