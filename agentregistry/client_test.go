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

package agentregistry

import (
	"context"
	"errors"
	"iter"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestClientList(t *testing.T) {
	tests := []struct {
		name         string
		respBody     string
		status       int // 0 => 200 OK
		opts         []ListOption
		wantPath     string
		wantFilter   string
		wantPageSize string
		list         func(context.Context, *Client, ...ListOption) (names []string, next string, err error)
		wantNames    []string
		wantNext     string
		wantErrCode  int // non-zero => expect an *APIError with this status
	}{
		{
			name:         "agents",
			respBody:     `{"agents":[{"displayName":"Foo"},{"displayName":"Bar"}],"nextPageToken":"n"}`,
			opts:         []ListOption{WithFilter("type=A2A"), WithPageSize(2)},
			wantPath:     "/projects/p/locations/l/agents",
			wantFilter:   "type=A2A",
			wantPageSize: "2",
			list: func(ctx context.Context, c *Client, o ...ListOption) ([]string, string, error) {
				p, err := c.ListAgents(ctx, o...)
				if err != nil {
					return nil, "", err
				}
				return mapNames(p.Agents, func(a Agent) string { return a.DisplayName }), p.NextPageToken, nil
			},
			wantNames: []string{"Foo", "Bar"},
			wantNext:  "n",
		},
		{
			name:     "mcpServers",
			respBody: `{"mcpServers":[{"displayName":"Data","mcpServerId":"data-mcp"}]}`,
			wantPath: "/projects/p/locations/l/mcpServers",
			list: func(ctx context.Context, c *Client, o ...ListOption) ([]string, string, error) {
				p, err := c.ListMCPServers(ctx, o...)
				if err != nil {
					return nil, "", err
				}
				return mapNames(p.MCPServers, func(s MCPServer) string { return s.DisplayName }), p.NextPageToken, nil
			},
			wantNames: []string{"Data"},
		},
		{
			name:     "endpoints",
			respBody: `{"endpoints":[{"displayName":"Gemini"}]}`,
			wantPath: "/projects/p/locations/l/endpoints",
			list: func(ctx context.Context, c *Client, o ...ListOption) ([]string, string, error) {
				p, err := c.ListEndpoints(ctx, o...)
				if err != nil {
					return nil, "", err
				}
				return mapNames(p.Endpoints, func(e Endpoint) string { return e.DisplayName }), p.NextPageToken, nil
			},
			wantNames: []string{"Gemini"},
		},
		{
			// A non-2xx status is surfaced as a typed *APIError.
			name:     "error status",
			respBody: `{"error":"boom"}`,
			status:   http.StatusServiceUnavailable,
			list: func(ctx context.Context, c *Client, o ...ListOption) ([]string, string, error) {
				p, err := c.ListAgents(ctx, o...)
				if err != nil {
					return nil, "", err
				}
				return mapNames(p.Agents, func(a Agent) string { return a.DisplayName }), p.NextPageToken, nil
			},
			wantErrCode: http.StatusServiceUnavailable,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var gotPath, gotFilter, gotPageSize string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				gotFilter = r.URL.Query().Get("filter")
				gotPageSize = r.URL.Query().Get("pageSize")
				if tc.status != 0 {
					w.WriteHeader(tc.status)
				}
				_, _ = w.Write([]byte(tc.respBody))
			}))
			defer srv.Close()

			names, next, err := tc.list(context.Background(), newTestClient(srv), tc.opts...)
			if tc.wantErrCode != 0 {
				var apiErr *APIError
				if !errors.As(err, &apiErr) {
					t.Fatalf("error = %v, want *APIError", err)
				}
				if apiErr.StatusCode != tc.wantErrCode {
					t.Errorf("APIError.StatusCode = %d, want %d", apiErr.StatusCode, tc.wantErrCode)
				}
				return
			}
			if err != nil {
				t.Fatalf("list error = %v", err)
			}
			if gotPath != tc.wantPath {
				t.Errorf("request path = %q, want %q", gotPath, tc.wantPath)
			}
			if gotFilter != tc.wantFilter {
				t.Errorf("filter param = %q, want %q", gotFilter, tc.wantFilter)
			}
			if gotPageSize != tc.wantPageSize {
				t.Errorf("pageSize param = %q, want %q", gotPageSize, tc.wantPageSize)
			}
			if !slices.Equal(names, tc.wantNames) {
				t.Errorf("names = %v, want %v", names, tc.wantNames)
			}
			if next != tc.wantNext {
				t.Errorf("nextPageToken = %q, want %q", next, tc.wantNext)
			}
		})
	}
}

func TestClientGet(t *testing.T) {
	tests := []struct {
		name         string
		respBody     string
		resourceName string
		wantPath     string
		// get fetches the resource and returns its display name plus one field
		// unique to that resource type (Description/MCPServerID/EndpointID),
		// which confirms Get decoded the correct concrete type.
		get           func(context.Context, *Client, string) (display, typeField string, err error)
		wantDisplay   string
		wantTypeField string
	}{
		{
			name:         "agent",
			respBody:     `{"displayName":"Summarizer","description":"sums things up"}`,
			resourceName: "projects/p/locations/l/agents/summarizer",
			wantPath:     "/projects/p/locations/l/agents/summarizer",
			get: func(ctx context.Context, c *Client, name string) (string, string, error) {
				a, err := c.GetAgent(ctx, name)
				if err != nil {
					return "", "", err
				}
				return a.DisplayName, a.Description, nil
			},
			wantDisplay:   "Summarizer",
			wantTypeField: "sums things up",
		},
		{
			name:         "mcpServer",
			respBody:     `{"displayName":"Data","mcpServerId":"data-mcp"}`,
			resourceName: "projects/p/locations/l/mcpServers/data-mcp",
			wantPath:     "/projects/p/locations/l/mcpServers/data-mcp",
			get: func(ctx context.Context, c *Client, name string) (string, string, error) {
				s, err := c.GetMCPServer(ctx, name)
				if err != nil {
					return "", "", err
				}
				return s.DisplayName, s.MCPServerID, nil
			},
			wantDisplay:   "Data",
			wantTypeField: "data-mcp",
		},
		{
			name:         "endpoint",
			respBody:     `{"displayName":"Gemini","endpointId":"gemini"}`,
			resourceName: "projects/p/locations/l/endpoints/gemini",
			wantPath:     "/projects/p/locations/l/endpoints/gemini",
			get: func(ctx context.Context, c *Client, name string) (string, string, error) {
				e, err := c.GetEndpoint(ctx, name)
				if err != nil {
					return "", "", err
				}
				return e.DisplayName, e.EndpointID, nil
			},
			wantDisplay:   "Gemini",
			wantTypeField: "gemini",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var gotPath string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				_, _ = w.Write([]byte(tc.respBody))
			}))
			defer srv.Close()

			display, typeField, err := tc.get(context.Background(), newTestClient(srv), tc.resourceName)
			if err != nil {
				t.Fatalf("get error = %v", err)
			}
			// The full resource name is used verbatim, not re-prefixed with the parent.
			if gotPath != tc.wantPath {
				t.Errorf("request path = %q, want %q", gotPath, tc.wantPath)
			}
			if display != tc.wantDisplay {
				t.Errorf("displayName = %q, want %q", display, tc.wantDisplay)
			}
			if typeField != tc.wantTypeField {
				t.Errorf("type-specific field = %q, want %q", typeField, tc.wantTypeField)
			}
		})
	}
}

// TestGetMCPServerDecodesFullResource guards the "output only" MCPServer
// metadata (tools[], annotations, timestamps, attributes) that the leaner
// list/get tests don't exercise, so it can't be silently dropped again.
func TestGetMCPServerDecodesFullResource(t *testing.T) {
	tests := []struct {
		name     string
		respBody string
		want     *MCPServer
	}{
		{
			name: "tools annotations timestamps and attributes",
			respBody: `{
				"name": "projects/p/locations/l/mcpServers/data",
				"mcpServerId": "data-mcp",
				"displayName": "Data",
				"tools": [
					{
						"name": "query",
						"description": "run a query",
						"annotations": {
							"title": "Query",
							"readOnlyHint": true,
							"destructiveHint": false,
							"idempotentHint": true,
							"openWorldHint": true
						}
					},
					{"name": "wipe", "description": "delete everything"}
				],
				"createTime": "2026-01-02T15:04:05Z",
				"updateTime": "2026-01-03T15:04:05Z",
				"attributes": {"agentregistry.googleapis.com/system/RuntimeReference": {"uri": "//gke/dep"}}
			}`,
			want: &MCPServer{
				Name:        "projects/p/locations/l/mcpServers/data",
				MCPServerID: "data-mcp",
				DisplayName: "Data",
				Tools: []Tool{
					{
						Name:        "query",
						Description: "run a query",
						Annotations: &Annotations{
							Title:          "Query",
							ReadOnlyHint:   true,
							IdempotentHint: true,
							OpenWorldHint:  true,
						},
					},
					{Name: "wipe", Description: "delete everything"},
				},
				CreateTime: "2026-01-02T15:04:05Z",
				UpdateTime: "2026-01-03T15:04:05Z",
				Attributes: map[string]any{
					"agentregistry.googleapis.com/system/RuntimeReference": map[string]any{"uri": "//gke/dep"},
				},
			},
		},
		{
			name:     "no tools reported leaves Tools nil",
			respBody: `{"displayName":"Data","mcpServerId":"data-mcp"}`,
			want:     &MCPServer{DisplayName: "Data", MCPServerID: "data-mcp"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = w.Write([]byte(tc.respBody))
			}))
			defer srv.Close()

			got, err := newTestClient(srv).GetMCPServer(t.Context(), "projects/p/locations/l/mcpServers/data")
			if err != nil {
				t.Fatalf("GetMCPServer() error = %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("GetMCPServer() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestClientAll(t *testing.T) {
	// twoPageAgents serves a1,a2 (nextPageToken tok2) then a3.
	twoPageAgents := func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Query().Get("pageToken") {
		case "":
			_, _ = w.Write([]byte(`{"agents":[{"displayName":"a1"},{"displayName":"a2"}],"nextPageToken":"tok2"}`))
		case "tok2":
			_, _ = w.Write([]byte(`{"agents":[{"displayName":"a3"}]}`))
		default:
			w.WriteHeader(http.StatusInternalServerError)
		}
	}
	agentsAll := func(ctx context.Context, c *Client) iter.Seq2[string, error] {
		return nameSeq(c.AllAgents(ctx), func(a *Agent) string { return a.DisplayName })
	}

	tests := []struct {
		name         string
		handler      http.HandlerFunc
		all          func(context.Context, *Client) iter.Seq2[string, error]
		stopAfter    int
		wantNames    []string
		wantErr      bool
		wantRequests int
	}{
		{
			name:         "agents paginate across pages",
			handler:      twoPageAgents,
			all:          agentsAll,
			wantNames:    []string{"a1", "a2", "a3"},
			wantRequests: 2,
		},
		{
			name:         "agents early stop skips the next page",
			handler:      twoPageAgents,
			all:          agentsAll,
			stopAfter:    1,
			wantNames:    []string{"a1"},
			wantRequests: 1,
		},
		{
			name: "agents error stops iteration",
			handler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = w.Write([]byte(`{"error":"boom"}`))
			},
			all:          agentsAll,
			wantErr:      true,
			wantRequests: 1,
		},
		{
			// The paginator must forward opts (filter) on *every* page, not just
			// the first. The handler 500s any request missing the filter, so if
			// the token page dropped it, page 2 would error instead of yielding a2.
			name: "filter is forwarded on every page",
			handler: func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Query().Get("filter") != "type=A2A" {
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(`{"error":"missing filter"}`))
					return
				}
				switch r.URL.Query().Get("pageToken") {
				case "":
					_, _ = w.Write([]byte(`{"agents":[{"displayName":"a1"}],"nextPageToken":"tok2"}`))
				case "tok2":
					_, _ = w.Write([]byte(`{"agents":[{"displayName":"a2"}]}`))
				default:
					w.WriteHeader(http.StatusInternalServerError)
				}
			},
			all: func(ctx context.Context, c *Client) iter.Seq2[string, error] {
				return nameSeq(c.AllAgents(ctx, WithFilter("type=A2A")), func(a *Agent) string { return a.DisplayName })
			},
			wantNames:    []string{"a1", "a2"},
			wantRequests: 2,
		},
		{
			name: "mcpServers single page",
			handler: func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write([]byte(`{"mcpServers":[{"displayName":"one"},{"displayName":"two"}]}`))
			},
			all: func(ctx context.Context, c *Client) iter.Seq2[string, error] {
				return nameSeq(c.AllMCPServers(ctx), func(s *MCPServer) string { return s.DisplayName })
			},
			wantNames: []string{"one", "two"},
		},
		{
			name: "endpoints single page",
			handler: func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write([]byte(`{"endpoints":[{"displayName":"gemini"}]}`))
			},
			all: func(ctx context.Context, c *Client) iter.Seq2[string, error] {
				return nameSeq(c.AllEndpoints(ctx), func(e *Endpoint) string { return e.DisplayName })
			},
			wantNames: []string{"gemini"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var requests int
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests++
				tc.handler(w, r)
			}))
			defer srv.Close()

			var got []string
			var gotErr error
			for name, err := range tc.all(context.Background(), newTestClient(srv)) {
				if err != nil {
					gotErr = err
					continue // let the iterator terminate on its own
				}
				got = append(got, name)
				if tc.stopAfter > 0 && len(got) >= tc.stopAfter {
					break
				}
			}

			if tc.wantErr && gotErr == nil {
				t.Errorf("iterator error = nil, want an error")
			}
			if !tc.wantErr && gotErr != nil {
				t.Fatalf("iterator error = %v, want nil", gotErr)
			}
			if tc.wantNames != nil && !slices.Equal(got, tc.wantNames) {
				t.Errorf("collected %v, want %v", got, tc.wantNames)
			}
			if tc.wantRequests > 0 && requests != tc.wantRequests {
				t.Errorf("server requests = %d, want %d", requests, tc.wantRequests)
			}
		})
	}
}

// --- test helpers ---

// newTestClient returns a Client whose requester talks to srv, exercising the
// full method -> getResource -> restRequester -> HTTP path.
func newTestClient(srv *httptest.Server) *Client {
	return &Client{requester: newTestRequester(srv), httpClient: srv.Client()}
}

// mapNames projects a display-name-like field out of each element.
func mapNames[T any](s []T, name func(T) string) []string {
	out := make([]string, len(s))
	for i, v := range s {
		out[i] = name(v)
	}
	return out
}

// nameSeq adapts a typed All* iterator to yield one name per item, so paging
// behavior can be exercised table-driven across resource types.
func nameSeq[T any](seq iter.Seq2[*T, error], name func(*T) string) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		for v, err := range seq {
			var s string
			if v != nil {
				s = name(v)
			}
			if !yield(s, err) {
				return
			}
		}
	}
}
