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
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestListAgents(t *testing.T) {
	var gotPath, gotFilter string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotFilter = r.URL.Query().Get("filter")
		_, _ = w.Write([]byte(`{"agents":[{"displayName":"Foo"},{"displayName":"Bar"}],"nextPageToken":"n"}`))
	}))
	defer srv.Close()

	page, err := newTestClient(srv).ListAgents(context.Background(), WithFilter("type=A2A"), WithPageSize(2))
	if err != nil {
		t.Fatalf("ListAgents() error = %v", err)
	}
	if want := "/projects/p/locations/l/agents"; gotPath != want {
		t.Errorf("request path = %q, want %q", gotPath, want)
	}
	if gotFilter != "type=A2A" {
		t.Errorf("filter param = %q, want type=A2A", gotFilter)
	}
	if len(page.Agents) != 2 || page.Agents[0].DisplayName != "Foo" || page.NextPageToken != "n" {
		t.Errorf("page = %+v, want two agents and nextPageToken n", page)
	}
}

func TestGetAgent(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_, _ = w.Write([]byte(`{"displayName":"Summarizer","description":"sums things up"}`))
	}))
	defer srv.Close()

	name := "projects/p/locations/l/agents/summarizer"
	got, err := newTestClient(srv).GetAgent(context.Background(), name)
	if err != nil {
		t.Fatalf("GetAgent() error = %v", err)
	}
	if want := "/" + name; gotPath != want {
		t.Errorf("request path = %q, want %q (full resource name, not re-prefixed)", gotPath, want)
	}
	if got.DisplayName != "Summarizer" || got.Description != "sums things up" {
		t.Errorf("agent = %+v, want Summarizer", got)
	}
}

func TestAllAgents_Paginates(t *testing.T) {
	var requests int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		switch tok := r.URL.Query().Get("pageToken"); tok {
		case "":
			_, _ = w.Write([]byte(`{"agents":[{"displayName":"a1"},{"displayName":"a2"}],"nextPageToken":"tok2"}`))
		case "tok2":
			_, _ = w.Write([]byte(`{"agents":[{"displayName":"a3"}]}`))
		default:
			t.Errorf("unexpected pageToken %q", tok)
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer srv.Close()

	var got []string
	for agent, err := range newTestClient(srv).AllAgents(context.Background()) {
		if err != nil {
			t.Fatalf("AllAgents() yielded error = %v", err)
		}
		got = append(got, agent.DisplayName)
	}

	want := []string{"a1", "a2", "a3"}
	if len(got) != len(want) {
		t.Fatalf("collected %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("item %d = %q, want %q", i, got[i], want[i])
		}
	}
	if requests != 2 {
		t.Errorf("server received %d requests, want 2 (one per page)", requests)
	}
}

func TestAllAgents_EarlyStop(t *testing.T) {
	var requests int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		_, _ = w.Write([]byte(`{"agents":[{"displayName":"a1"},{"displayName":"a2"}],"nextPageToken":"tok2"}`))
	}))
	defer srv.Close()

	var seen int
	for agent, err := range newTestClient(srv).AllAgents(context.Background()) {
		if err != nil {
			t.Fatalf("AllAgents() yielded error = %v", err)
		}
		seen++
		_ = agent
		break // stop after the first item
	}

	if seen != 1 {
		t.Errorf("consumed %d items before break, want 1", seen)
	}
	if requests != 1 {
		t.Errorf("server received %d requests, want 1 (must not fetch the next page after break)", requests)
	}
}

func TestAllAgents_ErrorStops(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"boom"}`))
	}))
	defer srv.Close()

	var iterations int
	var gotErr error
	for agent, err := range newTestClient(srv).AllAgents(context.Background()) {
		iterations++
		gotErr = err
		if agent != nil {
			t.Errorf("expected nil agent alongside error, got %+v", agent)
		}
	}
	if iterations != 1 {
		t.Fatalf("iterations = %d, want exactly 1 (error then stop)", iterations)
	}
	if gotErr == nil {
		t.Error("iterator error = nil, want the API error")
	}
}
