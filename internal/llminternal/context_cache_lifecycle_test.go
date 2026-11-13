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

package llminternal

import (
	"context"
	"errors"
	"iter"
	"net/http"
	"reflect"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

type eventList []*session.Event

func (e eventList) All() iter.Seq[*session.Event] { return slices.Values(e) }
func (e eventList) Len() int                      { return len(e) }
func (e eventList) At(i int) *session.Event       { return e[i] }

// stored returns a loader that yields events, as a session store would.
func stored(events session.Events) func(context.Context) (session.Events, error) {
	return func(context.Context) (session.Events, error) { return events, nil }
}

// record returns an event of author on branch recording the named cache, or
// fingerprint-only metadata when name is empty.
func record(author, branch, name string) *session.Event {
	m := &model.CacheMetadata{Fingerprint: "fp", ContentsCount: 1}
	if name != "" {
		m.CacheName = name
		m.InvocationsUsed = 1
		m.ExpireTime = time.Unix(3600, 0)
	}
	return &session.Event{Author: author, Branch: branch, LLMResponse: model.LLMResponse{CacheMetadata: m}}
}

func TestContextCacheLedger_Settle(t *testing.T) {
	tests := []struct {
		name        string
		tracked     []string
		events      eventList
		wantDeleted []string
	}{
		{
			name:        "nothing stored deletes everything tracked",
			tracked:     []string{"c/2", "c/1"},
			wantDeleted: []string{"c/1", "c/2"},
		},
		{
			name:        "the cache the latest record names is kept",
			tracked:     []string{"c/1", "c/2"},
			events:      eventList{record("a", "b", "c/1"), record("a", "b", "c/2")},
			wantDeleted: []string{"c/1"},
		},
		{
			name:        "a fingerprint-only latest record keeps nothing",
			tracked:     []string{"c/1"},
			events:      eventList{record("a", "b", "c/1"), record("a", "b", "")},
			wantDeleted: []string{"c/1"},
		},
		{
			name:    "each agent's latest record counts",
			tracked: []string{"c/1", "c/2"},
			events:  eventList{record("a", "b", "c/1"), record("other", "b", "c/2")},
		},
		{
			name:    "an agent on another branch is another agent",
			tracked: []string{"c/1", "c/2"},
			events:  eventList{record("a", "b", "c/1"), record("a", "b@2", "c/2")},
		},
		{
			name:        "events without cache state are skipped",
			tracked:     []string{"c/1"},
			events:      eventList{record("a", "b", "c/1"), {Author: "a", Branch: "b"}},
			wantDeleted: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ledger := NewContextCacheLedger()
			var deleted []string
			for _, name := range tt.tracked {
				ledger.Track(name, func(context.Context) { deleted = append(deleted, name) })
			}

			ledger.Settle(t.Context(), stored(tt.events))

			if diff := cmp.Diff(tt.wantDeleted, deleted); diff != "" {
				t.Errorf("deleted mismatch (-want +got):\n%s", diff)
			}
			deleted = nil
			ledger.Settle(t.Context(), stored(nil))
			if len(deleted) != 0 {
				t.Errorf("a second Settle deleted %v, want the ledger emptied by the first", deleted)
			}
		})
	}
}

// TestContextCacheLedger_SettleLoading checks when Settle reads the stored
// events, and what it does when it cannot.
func TestContextCacheLedger_SettleLoading(t *testing.T) {
	t.Run("an empty ledger reads nothing", func(t *testing.T) {
		ledger := NewContextCacheLedger()
		ledger.Settle(t.Context(), func(context.Context) (session.Events, error) {
			t.Error("Settle read the stored events with nothing tracked")
			return nil, nil
		})
	})

	t.Run("a failed read deletes nothing and empties the ledger", func(t *testing.T) {
		ledger := NewContextCacheLedger()
		var deleted []string
		for _, name := range []string{"c/1", "c/2"} {
			ledger.Track(name, func(context.Context) { deleted = append(deleted, name) })
		}

		ledger.Settle(t.Context(), func(context.Context) (session.Events, error) {
			return nil, errors.New("store unavailable")
		})
		if len(deleted) != 0 {
			t.Errorf("deleted %v when the stored events could not be read, want nothing", deleted)
		}

		ledger.Settle(t.Context(), stored(nil))
		if len(deleted) != 0 {
			t.Errorf("a later Settle deleted %v, want the ledger emptied by the failed one", deleted)
		}
	})
}

// TestContextCacheLedger_ConcurrentTrack checks that parallel sub-agents
// sharing an invocation can track caches at once.
func TestContextCacheLedger_ConcurrentTrack(t *testing.T) {
	ledger := NewContextCacheLedger()
	var mu sync.Mutex
	deleted := 0
	var wg sync.WaitGroup
	for i := range 20 {
		wg.Go(func() {
			name := "c/" + string(rune('a'+i))
			ledger.Track(name, func(context.Context) {
				mu.Lock()
				deleted++
				mu.Unlock()
			})
		})
	}
	wg.Wait()

	ledger.Settle(t.Context(), stored(nil))

	if deleted != 20 {
		t.Errorf("deleted %d caches, want all 20 tracked", deleted)
	}
}

func TestContextCacheLedger_NilIsInert(t *testing.T) {
	var ledger *ContextCacheLedger
	ledger.Track("c/1", func(context.Context) { t.Error("a nil ledger deleted a cache") })
	ledger.Settle(t.Context(), stored(eventList{}))
}

// TestCloneContextCacheConfig checks that writing to every mutable part of the
// copy's HTTP options leaves the original unchanged.
func TestCloneContextCacheConfig(t *testing.T) {
	// newConfig builds the config from values of its own each time, so that
	// want shares no pointer with original: a write through a pointer the
	// clone wrongly shares with original must not change want too.
	newConfig := func() *model.ContextCacheConfig {
		timeout := time.Second
		attempts := int32(3)
		initialDelay, maxDelay, expBase, jitter := 1.0, 60.0, 2.0, 1.0
		return &model.ContextCacheConfig{
			CacheIntervals: 4,
			CreateHTTPOptions: &genai.HTTPOptions{
				Headers: http.Header{"X-A": []string{"1"}},
				ExtraBody: map[string]any{
					"k":      "v",
					"labels": map[string]any{"team": "a"},
					"typed":  map[string]string{"x": "1"},
					"list":   []any{map[string]any{"i": 1}, "s"},
					"names":  []string{"n1"},
					"empty":  map[string]any{},
					"nilmap": map[string]any(nil),
				},
				Timeout: &timeout,
				RetryOptions: &genai.HTTPRetryOptions{
					Attempts:        &attempts,
					InitialDelay:    &initialDelay,
					MaxDelay:        &maxDelay,
					ExpBase:         &expBase,
					Jitter:          &jitter,
					HTTPStatusCodes: []int32{429},
				},
			},
		}
	}
	original := newConfig()
	want := newConfig()

	clone := CloneContextCacheConfig(original)
	clone.CacheIntervals = 9
	o := clone.CreateHTTPOptions
	o.Headers.Set("X-A", "changed")
	o.Headers.Set("X-B", "added")
	o.ExtraBody["k"] = "changed"
	// A callback writing to a map nested in the extra body, the case a copy of
	// only its top level leaves shared.
	o.ExtraBody["labels"].(map[string]any)["team"] = "changed"
	o.ExtraBody["labels"].(map[string]any)["added"] = "x"
	o.ExtraBody["typed"].(map[string]string)["x"] = "changed"
	o.ExtraBody["list"].([]any)[0].(map[string]any)["i"] = 99
	o.ExtraBody["list"].([]any)[1] = "changed"
	o.ExtraBody["names"].([]string)[0] = "changed"
	o.ExtraBody["empty"].(map[string]any)["added"] = "x"
	*o.Timeout = time.Hour
	*o.RetryOptions.Attempts = 9
	*o.RetryOptions.InitialDelay = 9
	*o.RetryOptions.MaxDelay = 9
	*o.RetryOptions.ExpBase = 9
	*o.RetryOptions.Jitter = 9
	o.RetryOptions.HTTPStatusCodes[0] = 500

	if diff := cmp.Diff(want, original, cmp.Comparer(func(a, b genai.ExtrasRequestProvider) bool { return true })); diff != "" {
		t.Errorf("original changed through its clone (-want +got):\n%s", diff)
	}
	if CloneContextCacheConfig(nil) != nil {
		t.Error("CloneContextCacheConfig(nil) != nil")
	}
}

// TestCloneContextCacheConfig_KeepsNilness checks that a nil map in the extra
// body, and a config without options, stay nil rather than becoming empty, so
// the copy sends the same request the original does.
func TestCloneContextCacheConfig_KeepsNilness(t *testing.T) {
	clone := CloneContextCacheConfig(&model.ContextCacheConfig{
		CreateHTTPOptions: &genai.HTTPOptions{ExtraBody: map[string]any{"nilmap": map[string]any(nil), "nilslice": []any(nil)}},
	})
	body := clone.CreateHTTPOptions.ExtraBody
	if m, ok := body["nilmap"].(map[string]any); !ok || m != nil {
		t.Errorf("nilmap = %#v, want a nil map", body["nilmap"])
	}
	if s, ok := body["nilslice"].([]any); !ok || s != nil {
		t.Errorf("nilslice = %#v, want a nil slice", body["nilslice"])
	}
	if got := CloneContextCacheConfig(&model.ContextCacheConfig{}).CreateHTTPOptions; got != nil {
		t.Errorf("options of a config without any = %v, want nil", got)
	}
	if got := CloneContextCacheConfig(&model.ContextCacheConfig{CreateHTTPOptions: &genai.HTTPOptions{}}).CreateHTTPOptions.ExtraBody; got != nil {
		t.Errorf("ExtraBody of options without one = %v, want nil", got)
	}
}

// TestCloneHTTPOptionsCoversEveryField fails when genai gives HTTPOptions or
// HTTPRetryOptions a field that refers to shared state, a pointer, map, slice,
// interface or function, that the copy in cloneHTTPOptions does not handle. A
// new such field would otherwise be shared between the caller and every
// request until someone noticed. Add the field to cloneHTTPOptions, then to
// the list here.
func TestCloneHTTPOptionsCoversEveryField(t *testing.T) {
	// handled lists the reference-valued fields cloneHTTPOptions copies, or
	// shares on purpose, as its doc comment says.
	handled := map[string]map[string]bool{
		"HTTPOptions": {
			"Headers":               true,
			"ExtraBody":             true,
			"Timeout":               true,
			"RetryOptions":          true,
			"ExtrasRequestProvider": true, // a function, shared as it is
		},
		"HTTPRetryOptions": {
			"Attempts":        true,
			"InitialDelay":    true,
			"MaxDelay":        true,
			"ExpBase":         true,
			"Jitter":          true,
			"HTTPStatusCodes": true,
		},
	}
	for _, typ := range []reflect.Type{reflect.TypeFor[genai.HTTPOptions](), reflect.TypeFor[genai.HTTPRetryOptions]()} {
		for field := range typ.Fields() {
			switch field.Type.Kind() {
			case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Interface, reflect.Func, reflect.Chan:
				if !handled[typ.Name()][field.Name] {
					t.Errorf("%s.%s is a %s that cloneHTTPOptions does not copy, so the caller and every request would share it", typ.Name(), field.Name, field.Type.Kind())
				}
			}
		}
	}
}

func TestInheritModelFields(t *testing.T) {
	usage := &genai.GenerateContentResponseUsageMetadata{PromptTokenCount: 5000}
	cache := &model.CacheMetadata{CacheName: "c/1", Fingerprint: "fp"}
	ownCache := &model.CacheMetadata{Fingerprint: "own"}
	text := genai.NewContentFromText("rewritten", genai.RoleModel)

	tests := []struct {
		name        string
		original    *model.LLMResponse
		replacement *model.LLMResponse
		want        *model.LLMResponse
	}{
		{
			name:        "cache state and usage are carried onto a replacement without them",
			original:    &model.LLMResponse{UsageMetadata: usage, CacheMetadata: cache},
			replacement: &model.LLMResponse{Content: text},
			want:        &model.LLMResponse{Content: text, UsageMetadata: usage, CacheMetadata: cache},
		},
		{
			name:        "a replacement's own cache state is kept",
			original:    &model.LLMResponse{CacheMetadata: cache},
			replacement: &model.LLMResponse{Content: text, CacheMetadata: ownCache},
			want:        &model.LLMResponse{Content: text, CacheMetadata: ownCache},
		},
		{
			name:        "nothing to carry",
			original:    &model.LLMResponse{},
			replacement: &model.LLMResponse{Content: text},
			want:        &model.LLMResponse{Content: text},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := *tt.replacement

			got := inheritModelFields(tt.original, tt.replacement)

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("inheritModelFields() mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(&before, tt.replacement); diff != "" {
				t.Errorf("the callback's replacement was modified (-before +after):\n%s", diff)
			}
		})
	}
}
