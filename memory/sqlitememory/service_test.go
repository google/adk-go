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

package sqlitememory_test

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/memory/sqlitememory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/preloadmemorytool"
)

func TestPersistenceAndScope(t *testing.T) {
	cfg := sqlitememory.Config{Path: filepath.Join(t.TempDir(), "memory#%.db")}
	svc := openService(t, cfg)
	event := textEvent("original", "I prefer 乌龙茶 on cold mornings.")
	event.CustomMetadata = map[string]any{"source": "test"}
	sess := newSession(t, "app", "user", "session", event)
	importSession(t, svc, sess)
	importSession(t, svc, sess)
	importSession(t, svc, newSession(t, "app", "user", "other-session", textEvent("other-session", "乌龙茶")))
	importSession(t, svc, newSession(t, "another-app", "user", "session", textEvent("other-app", "乌龙茶")))
	importSession(t, svc, newSession(t, "app", "another-user", "session", textEvent("other-user", "乌龙茶")))
	if err := svc.Close(); err != nil {
		t.Fatalf("Close() failed: %v", err)
	}
	reopened := openService(t, cfg)
	matches := search(t, reopened, "app", "user", "乌龙")
	checkIDs(t, matches, "original", "other-session")
	got := matches[0]
	if got.Author != event.Author || !got.Timestamp.Equal(event.Timestamp) || got.CustomMetadata["source"] != "test" ||
		got.Content == nil || len(got.Content.Parts) != 1 || got.Content.Role != event.Content.Role || got.Content.Parts[0].Text != event.Content.Parts[0].Text {
		t.Error("search did not preserve the event's content and metadata")
	}

	importSession(t, reopened, newSession(t, "app", "user", "session", textEvent("replacement", "I prefer coffee.")))
	checkIDs(t, search(t, reopened, "app", "user", "乌龙"), "other-session")
	checkIDs(t, search(t, reopened, "app", "user", "coffee"), "replacement")
	checkIDs(t, search(t, reopened, "another-app", "user", "乌龙"), "other-app")
	checkIDs(t, search(t, reopened, "app", "another-user", "乌龙"), "other-user")

	importSession(t, reopened, newSession(t, "app", "user", "session"))
	checkIDs(t, search(t, reopened, "app", "user", "coffee"))
}

func TestReimportChangesTieOrder(t *testing.T) {
	svc := openService(t, sqlitememory.Config{Path: filepath.Join(t.TempDir(), "memory.db")})
	first := newSession(t, "app", "user", "first", textEvent("first", "oolong tea"))
	second := newSession(t, "app", "user", "second", textEvent("second", "green tea"))
	importSession(t, svc, first)
	importSession(t, svc, second)
	checkIDs(t, search(t, svc, "app", "user", "tea"), "first", "second")

	importSession(t, svc, first)
	checkIDs(t, search(t, svc, "app", "user", "tea"), "second", "first")
}

func TestImportRollback(t *testing.T) {
	svc := openService(t, sqlitememory.Config{Path: filepath.Join(t.TempDir(), "memory.db")})
	importSession(t, svc, newSession(t, "app", "user", "session", textEvent("original", "original memory")))
	invalid := textEvent("invalid", "invalid memory")
	invalid.CustomMetadata = map[string]any{"unsupported": make(chan struct{})}
	replacement := newSession(t, "app", "user", "session", textEvent("replacement", "replacement memory"), invalid)
	if err := svc.AddSessionToMemory(t.Context(), replacement); err == nil {
		t.Fatal("AddSessionToMemory() succeeded with non-serializable metadata")
	}
	checkIDs(t, search(t, svc, "app", "user", "memory"), "original")
}

func TestKeywordSearch(t *testing.T) {
	svc := openService(t, sqlitememory.Config{Path: filepath.Join(t.TempDir(), "memory.db")})
	withThought := textEvent("with-thought", "Project Alpha")
	withThought.Content.Parts = append(withThought.Content.Parts, &genai.Part{Text: "hidden reasoning", Thought: true})
	thoughtOnly := textEvent("thought-only", "hidden reasoning")
	thoughtOnly.Content.Parts[0].Thought = true
	var events []*session.Event
	for i := range 12 {
		events = append(events, textEvent(fmt.Sprintf("project-%d", i), "project"))
	}
	events = append(events, thoughtOnly, withThought)
	importSession(t, svc, newSession(t, "app", "user", "session", events...))

	for _, query := range []string{"", "?!", "hidden", "pro"} {
		checkIDs(t, search(t, svc, "app", "user", query))
	}
	matches := search(t, svc, "app", "user", "PROJECT alpha")
	checkIDs(t, matches, "with-thought", "project-0", "project-1", "project-2", "project-3", "project-4", "project-5", "project-6", "project-7", "project-8")
	if content := matches[0].Content; content == nil || len(content.Parts) != 1 || content.Parts[0].Thought {
		t.Error("search result included thought content")
	}
}

func TestInvalidInput(t *testing.T) {
	if svc, err := sqlitememory.New(t.Context(), sqlitememory.Config{}); err == nil {
		_ = svc.Close()
		t.Fatal("New() succeeded with an empty path")
	} else if err.Error() != "sqlitememory: database path must not be empty" {
		t.Error("New() did not report that the path was empty")
	}
	if svc, err := sqlitememory.New(t.Context(), sqlitememory.Config{Path: t.TempDir()}); err == nil {
		_ = svc.Close()
		t.Fatal("New() succeeded with a directory as the database file")
	}
	svc := openService(t, sqlitememory.Config{Path: filepath.Join(t.TempDir(), "memory.db")})
	if err := svc.AddSessionToMemory(t.Context(), nil); err == nil {
		t.Error("AddSessionToMemory() succeeded with a nil session")
	}
	if _, err := svc.SearchMemory(t.Context(), nil); err == nil {
		t.Error("SearchMemory() succeeded with a nil request")
	}
}

func TestPreloadMemory(t *testing.T) {
	ctx := t.Context()
	cfg := sqlitememory.Config{Path: filepath.Join(t.TempDir(), "memory.db")}
	svc := openService(t, cfg)
	const memoryText = "My favorite tea is 乌龙茶."
	importSession(t, svc, newSession(t, "app", "user", "previous", textEvent("tea", memoryText)))
	llm := &testutil.MockModel{
		Responses: []*genai.Content{genai.NewContentFromText("ok", genai.RoleModel)},
	}
	a, err := llmagent.New(llmagent.Config{
		Name: "memory_agent", Model: llm,
		Tools: []tool.Tool{preloadmemorytool.New()},
	})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName: "app", Agent: a, SessionService: session.InMemoryService(),
		MemoryService: svc, AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New() failed: %v", err)
	}
	for _, err := range r.Run(ctx, "user", "current", genai.NewContentFromText("乌龙", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() failed: %v", err)
		}
	}
	if len(llm.Requests) != 1 {
		t.Fatalf("model received %d requests, want 1", len(llm.Requests))
	}
	config := llm.Requests[0].Config
	if config == nil || config.SystemInstruction == nil {
		t.Fatal("model request has no system instructions")
	}
	if !slices.ContainsFunc(config.SystemInstruction.Parts, func(part *genai.Part) bool {
		return part != nil && strings.Contains(part.Text, memoryText)
	}) {
		t.Error("model instructions did not include the imported memory")
	}
}

func openService(t *testing.T, cfg sqlitememory.Config) *sqlitememory.Service {
	t.Helper()
	svc, err := sqlitememory.New(t.Context(), cfg)
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	t.Cleanup(func() {
		if err := svc.Close(); err != nil {
			t.Errorf("Close() failed: %v", err)
		}
	})
	return svc
}

func textEvent(id, text string) *session.Event {
	return &session.Event{
		ID: id, Author: "user", Timestamp: time.Date(2026, time.January, 1, 12, 0, 0, 0, time.UTC),
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleUser)},
	}
}

func newSession(t *testing.T, app, user, id string, events ...*session.Event) session.Session {
	t.Helper()
	ctx := t.Context()
	svc := session.InMemoryService()
	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: app, UserID: user, SessionID: id})
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}
	for _, event := range events {
		if err := svc.AppendEvent(ctx, resp.Session, event); err != nil {
			t.Fatalf("AppendEvent() failed: %v", err)
		}
	}
	return resp.Session
}

func importSession(t *testing.T, svc memory.Service, sess session.Session) {
	t.Helper()
	if err := svc.AddSessionToMemory(t.Context(), sess); err != nil {
		t.Fatalf("AddSessionToMemory() failed: %v", err)
	}
}

func search(t *testing.T, svc memory.Service, app, user, query string) []memory.Entry {
	t.Helper()
	resp, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{AppName: app, UserID: user, Query: query})
	if err != nil {
		t.Fatalf("SearchMemory() failed: %v", err)
	}
	return resp.Memories
}

func checkIDs(t *testing.T, entries []memory.Entry, want ...string) {
	t.Helper()
	ids := make([]string, 0, len(entries))
	for _, entry := range entries {
		ids = append(ids, entry.ID)
	}
	if !slices.Equal(ids, want) {
		t.Fatalf("search returned event IDs %q, want %q", ids, want)
	}
}
