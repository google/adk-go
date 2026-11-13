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

package memory_test

import (
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
)

type memoryMetadataRecord struct {
	Values []string
	Time   time.Time
}

type memoryMetadataStrings []string

type memoryMetadataOtherStrings []string

func memoryCopyEvent() *session.Event {
	event := memoryTextEvent("event", "Python")
	event.Content.Parts = append(event.Content.Parts,
		&genai.Part{InlineData: &genai.Blob{Data: []byte{1, 2}, MIMEType: "image/png"}},
		&genai.Part{FunctionCall: &genai.FunctionCall{Args: map[string]any{
			"items": []any{map[string]any{"value": "original"}},
		}}},
	)
	event.Content.Parts[0].ThoughtSignature = []byte{3, 4}
	event.Content.Parts[0].PartMetadata = map[string]any{"values": []string{"original"}}
	values := []string{"original", "tail"}
	event.CustomMetadata = map[string]any{
		"label":         "original",
		"nested":        map[string]any{"items": []any{map[string]any{"value": "original"}}},
		"typed":         []map[string][]int{{"numbers": {1, 2}}},
		"record":        &memoryMetadataRecord{Values: []string{"original"}, Time: time.Unix(123, 456)},
		"array":         [1][]string{{"original"}},
		"array pointer": &[1][]string{{"original"}},
		"views":         map[string][]string{"short": values[:1], "long": values},
		"named":         []any{memoryMetadataStrings(values), memoryMetadataOtherStrings(values)},
	}
	return event
}

func searchCopyMemory(t *testing.T, svc memory.Service, query string) []memory.Entry {
	t.Helper()
	response, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{AppName: "app", UserID: "user", Query: query})
	if err != nil {
		t.Fatal("memory search failed")
	}
	return response.Memories
}

func TestInMemoryServiceCopiesData(t *testing.T) {
	mutations := []struct {
		name   string
		mutate func(*genai.Content, map[string]any)
	}{
		{"role", func(content *genai.Content, _ map[string]any) { content.Role = genai.RoleModel }},
		{"text", func(content *genai.Content, _ map[string]any) { content.Parts[0].Text = "Java" }},
		{"parts slice", func(content *genai.Content, _ map[string]any) { content.Parts[0] = genai.NewPartFromText("Java") }},
		{"inline data", func(content *genai.Content, _ map[string]any) { content.Parts[1].InlineData.Data[0] = 9 }},
		{"thought signature", func(content *genai.Content, _ map[string]any) { content.Parts[0].ThoughtSignature[0] = 9 }},
		{"part metadata", func(content *genai.Content, _ map[string]any) {
			content.Parts[0].PartMetadata["values"].([]string)[0] = "changed"
		}},
		{"function arguments", func(content *genai.Content, _ map[string]any) {
			content.Parts[2].FunctionCall.Args["items"].([]any)[0].(map[string]any)["value"] = "changed"
		}},
		{"metadata", func(_ *genai.Content, metadata map[string]any) { metadata["label"] = "changed" }},
		{"nested metadata", func(_ *genai.Content, metadata map[string]any) {
			metadata["nested"].(map[string]any)["items"].([]any)[0].(map[string]any)["value"] = "changed"
		}},
		{"typed metadata", func(_ *genai.Content, metadata map[string]any) {
			metadata["typed"].([]map[string][]int)[0]["numbers"][0] = 9
		}},
		{"struct metadata", func(_ *genai.Content, metadata map[string]any) {
			metadata["record"].(*memoryMetadataRecord).Values[0] = "changed"
		}},
		{"array metadata", func(_ *genai.Content, metadata map[string]any) {
			metadata["array"].([1][]string)[0][0] = "changed"
		}},
		{"array pointer metadata", func(_ *genai.Content, metadata map[string]any) {
			metadata["array pointer"].(*[1][]string)[0][0] = "changed"
		}},
		{"slice views", func(_ *genai.Content, metadata map[string]any) {
			metadata["views"].(map[string][]string)["long"][0] = "changed"
		}},
		{"named slices", func(_ *genai.Content, metadata map[string]any) {
			metadata["named"].([]any)[0].(memoryMetadataStrings)[0] = "changed"
		}},
	}
	for _, store := range []struct {
		name string
		add  func(*testing.T, memory.Service, string, ...*session.Event)
	}{
		{"session", addSession},
		{"events", addEvents},
	} {
		t.Run(store.name, func(t *testing.T) {
			for _, origin := range []string{"source", "result"} {
				t.Run(origin, func(t *testing.T) {
					for _, mutation := range mutations {
						t.Run(mutation.name, func(t *testing.T) {
							event := memoryCopyEvent()
							want := memoryCopyEvent()
							svc := memory.InMemoryService()
							store.add(t, svc, "session", event)
							first := searchCopyMemory(t, svc, "Python")
							if len(first) != 1 {
								t.Fatal("seed memory not found")
							}
							if !cmp.Equal(first[0].Content, want.Content) || !cmp.Equal(first[0].CustomMetadata, want.CustomMetadata) {
								t.Fatal("ingestion changed memory data or types")
							}
							if origin == "source" {
								mutation.mutate(event.Content, event.CustomMetadata)
							} else {
								mutation.mutate(first[0].Content, first[0].CustomMetadata)
							}
							again := searchCopyMemory(t, svc, "Python")
							if len(again) != 1 {
								t.Fatal("original keyword no longer matches")
							}
							if !cmp.Equal(again[0].Content, want.Content) {
								t.Error("external mutation changed stored content")
							}
							if !cmp.Equal(again[0].CustomMetadata, want.CustomMetadata) {
								t.Error("external mutation changed stored metadata")
							}
							if len(searchCopyMemory(t, svc, "Java")) != 0 {
								t.Error("external mutation changed the search index")
							}
						})
					}
				})
			}
		})
	}
}

func TestInMemoryServiceCopiesPreserveNilAndEmpty(t *testing.T) {
	for _, metadata := range []map[string]any{nil, {}, {
		"nil": nil, "map": map[string]any(nil), "slice": []string(nil), "pointer": (*memoryMetadataRecord)(nil),
		"empty map": map[string]any{}, "empty slice": []string{},
	}} {
		event := memoryTextEvent("event", "Python")
		event.CustomMetadata = metadata
		svc := memory.InMemoryService()
		if err := svc.AddSessionToMemory(t.Context(), makeSession(t, "app", "user", "session", []*session.Event{event})); err != nil {
			t.Fatal("add session failed")
		}
		got := searchCopyMemory(t, svc, "Python")
		if len(got) != 1 || !cmp.Equal(got[0].CustomMetadata, metadata) {
			t.Error("copy changed nil or empty metadata")
		}
	}
}

func TestInMemoryServiceCopiesDefinedPointerMetadata(t *testing.T) {
	type countPointer *int
	count := 7
	pointer := countPointer(&count)
	event := memoryTextEvent("event", "Python")
	event.CustomMetadata = map[string]any{
		"first":  pointer,
		"second": pointer,
		"nested": []any{pointer},
		"nil":    countPointer(nil),
	}
	svc := memory.InMemoryService()
	if err := svc.AddSessionToMemory(t.Context(), makeSession(t, "app", "user", "session", []*session.Event{event})); err != nil {
		t.Fatal("add session failed")
	}
	count = 8
	for range 2 {
		found := searchCopyMemory(t, svc, "Python")
		if len(found) != 1 {
			t.Fatal("seed memory not found")
		}
		metadata := found[0].CustomMetadata
		first, ok := metadata["first"].(countPointer)
		if !ok {
			t.Fatal("copy changed the defined pointer type")
		}
		second, ok := metadata["second"].(countPointer)
		if !ok {
			t.Fatal("copy changed the repeated pointer's type")
		}
		nested, ok := metadata["nested"].([]any)[0].(countPointer)
		if !ok {
			t.Fatal("copy changed the nested pointer's type")
		}
		if first != second || first != nested {
			t.Error("copy did not preserve shared pointers")
		}
		if *first != 7 {
			t.Error("external mutation changed stored metadata")
		}
		if nilPointer, ok := metadata["nil"].(countPointer); !ok || nilPointer != nil {
			t.Error("copy changed the typed nil pointer")
		}
		*first = 9
	}
}

func TestInMemoryServicePreservesOpaqueMetadata(t *testing.T) {
	type record struct {
		Name    string
		Builder strings.Builder
	}
	for _, tt := range []struct {
		name    string
		value   any
		builder func(any) *strings.Builder
	}{
		{
			name: "builder pointer", value: new(strings.Builder),
			builder: func(value any) *strings.Builder { return value.(*strings.Builder) },
		},
		{
			name: "builder in struct", value: new(record),
			builder: func(value any) *strings.Builder { return &value.(*record).Builder },
		},
		{
			name: "builder in array", value: new([1]strings.Builder),
			builder: func(value any) *strings.Builder { return &value.(*[1]strings.Builder)[0] },
		},
		{
			name: "slice of builders", value: make([]strings.Builder, 1),
			builder: func(value any) *strings.Builder { return &value.([]strings.Builder)[0] },
		},
		{
			name: "slice of structs", value: make([]record, 1),
			builder: func(value any) *strings.Builder { return &value.([]record)[0].Builder },
		},
		{
			name: "slice of arrays", value: make([][1]strings.Builder, 1),
			builder: func(value any) *strings.Builder { return &value.([][1]strings.Builder)[0][0] },
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			original := tt.builder(tt.value)
			original.WriteString("seed")
			event := memoryTextEvent("event", "Python")
			event.CustomMetadata = map[string]any{
				"value":  tt.value,
				"nested": []any{map[string]any{"value": tt.value}},
			}
			svc := memory.InMemoryService()
			if err := svc.AddSessionToMemory(t.Context(), makeSession(t, "app", "user", "session", []*session.Event{event})); err != nil {
				t.Fatal("add session failed")
			}
			original.WriteString(" updated")
			for range 2 {
				found := searchCopyMemory(t, svc, "Python")
				if len(found) != 1 {
					t.Fatal("seed memory not found")
				}
				metadata := found[0].CustomMetadata
				for _, value := range []any{metadata["value"], metadata["nested"].([]any)[0].(map[string]any)["value"]} {
					got := tt.builder(value)
					if got != original {
						t.Fatal("opaque builder was copied")
					}
					got.WriteString("x")
				}
			}
		})
	}
}

func TestInMemoryServiceCopiesOpaqueMetadataContainers(t *testing.T) {
	type record struct {
		Builder *strings.Builder
		Values  []string
	}
	source := &record{Builder: new(strings.Builder), Values: []string{"original"}}
	source.Builder.WriteString("seed")
	event := memoryTextEvent("event", "Python")
	event.CustomMetadata = map[string]any{
		"record":   source,
		"builders": []*strings.Builder{source.Builder},
	}
	svc := memory.InMemoryService()
	if err := svc.AddSessionToMemory(t.Context(), makeSession(t, "app", "user", "session", []*session.Event{event})); err != nil {
		t.Fatal("add session failed")
	}
	source.Values[0] = "source changed"
	for range 2 {
		found := searchCopyMemory(t, svc, "Python")
		if len(found) != 1 {
			t.Fatal("seed memory not found")
		}
		metadata := found[0].CustomMetadata
		got := metadata["record"].(*record)
		builders := metadata["builders"].([]*strings.Builder)
		if got.Builder != source.Builder || builders[0] != source.Builder {
			t.Error("opaque pointers in copied containers changed")
		}
		if got.Values[0] != "original" {
			t.Error("external mutation changed the copied container")
		}
		got.Values[0] = "result changed"
		builders[0] = nil
	}
}

func TestInMemoryServiceCopiesCyclicMetadata(t *testing.T) {
	type node struct {
		Value string
		Next  *node
	}
	n := &node{Value: "original"}
	n.Next = n
	slice := make([]any, 2)
	slice[0], slice[1] = slice, "original"
	metadata := map[string]any{"label": "original", "node": n, "slice": slice, "keyed": map[*node]string{n: "value"}}
	metadata["self"] = metadata
	event := memoryTextEvent("event", "Python")
	event.CustomMetadata = metadata
	svc := memory.InMemoryService()
	if err := svc.AddSessionToMemory(t.Context(), makeSession(t, "app", "user", "session", []*session.Event{event})); err != nil {
		t.Fatal("add session failed")
	}
	metadata["label"], n.Value, slice[1] = "source changed", "source changed", "source changed"
	for range 2 {
		found := searchCopyMemory(t, svc, "Python")
		if len(found) != 1 {
			t.Fatal("seed memory not found")
		}
		got := found[0].CustomMetadata
		gotNode := got["node"].(*node)
		gotSlice := got["slice"].([]any)
		if got["label"] != "original" || gotNode.Value != "original" || gotSlice[1] != "original" {
			t.Fatal("external mutation changed cyclic metadata")
		}
		if gotNode.Next != gotNode || got["keyed"].(map[*node]string)[gotNode] != "value" {
			t.Error("copy did not preserve shared pointers")
		}
		got["self"].(map[string]any)["label"] = "result changed"
		gotNode.Next.Value = "result changed"
		gotSlice[0].([]any)[1] = "result changed"
		if got["label"] != "result changed" || gotNode.Value != "result changed" || gotSlice[1] != "result changed" {
			t.Error("copy did not preserve cycles")
		}
	}
}

func TestInMemoryServiceIndependentResultsConcurrent(t *testing.T) {
	svc := memory.InMemoryService()
	if err := svc.AddSessionToMemory(t.Context(), makeSession(t, "app", "user", "session", []*session.Event{memoryCopyEvent()})); err != nil {
		t.Fatal("add session failed")
	}
	const workers = 8
	results := make([]memory.Entry, workers)
	for i := range results {
		found := searchCopyMemory(t, svc, "Python")
		if len(found) != 1 {
			t.Fatal("seed memory not found")
		}
		results[i] = found[0]
	}
	var wg sync.WaitGroup
	for i := range results {
		wg.Go(func() {
			for range 100 {
				results[i].Content.Parts[0].Text = "Java"
				results[i].CustomMetadata["typed"].([]map[string][]int)[0]["numbers"][0]++
				found := searchCopyMemory(t, svc, "Python")
				if len(found) != 1 || found[0].Content.Parts[0].Text != "Python" {
					t.Error("result mutation changed stored memory")
					return
				}
			}
		})
	}
	wg.Wait()
}
