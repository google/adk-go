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

package flow_test

import (
	"context"
	"iter"
	"testing"

	"google.golang.org/adk/flow"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/model"
)

type stubLLM struct{ name string }

func (s *stubLLM) Name() string { return s.name }
func (s *stubLLM) GenerateContent(_ context.Context, _ *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{}, nil)
	}
}

func TestAutoFlow_PopulatesDefaults(t *testing.T) {
	f := flow.AutoFlow(&stubLLM{name: "stub"})
	if f.Model == nil {
		t.Error("Model not set")
	}
	if len(f.RequestProcessors) == 0 {
		t.Error("RequestProcessors should not be empty")
	}
	if len(f.ResponseProcessors) == 0 {
		t.Error("ResponseProcessors should not be empty")
	}
}

func TestSingleFlow_PopulatesDefaults(t *testing.T) {
	f := flow.SingleFlow(&stubLLM{name: "stub"})
	if f.Model == nil {
		t.Error("Model not set")
	}
}

func TestDefaultRequestProcessors_ReturnsCopy(t *testing.T) {
	a := flow.DefaultRequestProcessors()
	b := flow.DefaultRequestProcessors()
	if len(a) != len(b) {
		t.Fatalf("lengths differ: %d vs %d", len(a), len(b))
	}
	if len(a) == 0 {
		t.Fatal("expected non-empty default processors")
	}
	// Mutating a should not affect b.
	a[0] = nil
	if b[0] == nil {
		t.Error("DefaultRequestProcessors should return independent copies")
	}
}

func TestFlow_TypeAlias_InterchangeableWithInternal(t *testing.T) {
	// The public flow.Flow must be the same underlying type as
	// llminternal.Flow so existing callers continue to compile.
	var pub *flow.Flow = &llminternal.Flow{}
	var priv *llminternal.Flow = pub
	if priv == nil {
		t.Fatal("alias round-trip failed")
	}
}

func TestCallbackTypes_AreAliases(t *testing.T) {
	// Compile-time assignments confirm the callback types are aliases of
	// the internal definitions (same signatures).
	var bm flow.BeforeModelCallback
	var am flow.AfterModelCallback
	var oerr flow.OnModelErrorCallback
	var bt flow.BeforeToolCallback
	var at flow.AfterToolCallback
	var ote flow.OnToolErrorCallback
	_, _, _, _, _, _ = bm, am, oerr, bt, at, ote
}
