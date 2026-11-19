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

package classifier

import (
	"context"
	"errors"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

type cloneErrorJSON struct{ err error }

func (v cloneErrorJSON) MarshalJSON() ([]byte, error) { return nil, v.err }

type cloneErrorModel struct{ called bool }

func (*cloneErrorModel) Name() string { return "clone-error-test" }

func (m *cloneErrorModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.called = true
	return func(func(*model.LLMResponse, error) bool) {}
}

func TestLLMClassifierConfigCloneErrors(t *testing.T) {
	wantErr := errors.New("cannot encode test value")
	cfg := &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{
		ExtraBody: map[string]any{"value": cloneErrorJSON{err: wantErr}},
	}}
	t.Run("constructor", func(t *testing.T) {
		m := &cloneErrorModel{}
		got, err := NewLLM(LLMConfig{Model: m, GenerateContentConfig: cfg})
		if !errors.Is(err, ErrInvalidLLMConfig) || !errors.Is(err, wantErr) {
			t.Fatal("NewLLM did not preserve the config error and its cause")
		}
		if got != nil {
			t.Fatal("NewLLM returned a classifier with an uncopyable config")
		}
	})
	t.Run("request", func(t *testing.T) {
		m := &cloneErrorModel{}
		// NewLLM rejects this config; inject it to exercise request-time errors.
		c := &LLMClassifier{llm: m, baseConfig: cfg}
		got, err := c.Classify(t.Context(), "input", map[string]Question{"answer": Binary{}})
		if !errors.Is(err, ErrInvalidLLMConfig) || !errors.Is(err, wantErr) {
			t.Fatal("Classify did not preserve the config error and its cause")
		}
		if got != nil {
			t.Fatal("Classify returned a response with an uncopyable config")
		}
		if m.called {
			t.Fatal("Classify called the model after copying failed")
		}
	})
}
