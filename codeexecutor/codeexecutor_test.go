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

package codeexecutor_test

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/codeexecutor"
)

var _ codeexecutor.BuiltIn = codeexecutor.NewBuiltInExecutor()

// pythonDefaults are the defaults of adk-python's BaseCodeExecutor
// (code_executors/base_code_executor.py:50-82).
func pythonDefaults() codeexecutor.Settings {
	return codeexecutor.Settings{
		OptimizeDataFile:          false,
		Stateful:                  false,
		ErrorRetryAttempts:        new(2),
		CodeBlockDelimiters:       [][2]string{{"```tool_code\n", "\n```"}, {"```python\n", "\n```"}},
		ExecutionResultDelimiters: &[2]string{"```tool_output\n", "\n```"},
		TimeoutSeconds:            nil,
	}
}

func TestDefaultSettingsAndBuiltInExecutor(t *testing.T) {
	sources := []struct {
		name     string
		settings func() codeexecutor.Settings
	}{
		{name: "DefaultSettings", settings: codeexecutor.DefaultSettings},
		{name: "NewBuiltInExecutor", settings: codeexecutor.NewBuiltInExecutor().Settings},
		{name: "zero BuiltInExecutor", settings: (&codeexecutor.BuiltInExecutor{}).Settings},
		{name: "nil BuiltInExecutor", settings: (*codeexecutor.BuiltInExecutor)(nil).Settings},
	}
	for _, src := range sources {
		t.Run(src.name, func(t *testing.T) {
			got := src.settings()
			if diff := cmp.Diff(pythonDefaults(), got); diff != "" {
				t.Fatalf("Settings mismatch (-want +got):\n%s", diff)
			}

			// A caller that edits the returned value changes no later call.
			got.CodeBlockDelimiters[0] = [2]string{"<a>", "</a>"}
			*got.ErrorRetryAttempts = 7
			got.ExecutionResultDelimiters[0] = "<out>"
			if diff := cmp.Diff(pythonDefaults(), src.settings()); diff != "" {
				t.Errorf("Settings after editing an earlier result mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("ExecuteCode returns no result", func(t *testing.T) {
		cancelled, cancel := context.WithCancel(t.Context())
		cancel()
		for _, ctx := range []context.Context{t.Context(), cancelled} {
			res, err := codeexecutor.NewBuiltInExecutor().ExecuteCode(ctx, nil, &codeexecutor.Input{Code: "print(1)"})
			if res != nil || err != nil {
				t.Errorf("ExecuteCode(ctx with Err() = %v) = (%+v, %v), want (nil, nil)", ctx.Err(), res, err)
			}
		}
	})
}

func TestSettingsWithDefaults(t *testing.T) {
	tests := []struct {
		name string
		in   codeexecutor.Settings
		want codeexecutor.Settings
	}{
		{
			name: "zero value",
			in:   codeexecutor.Settings{},
			want: pythonDefaults(),
		},
		{
			name: "only Stateful set",
			in:   codeexecutor.Settings{Stateful: true},
			want: codeexecutor.Settings{
				Stateful:                  true,
				ErrorRetryAttempts:        new(2),
				CodeBlockDelimiters:       [][2]string{{"```tool_code\n", "\n```"}, {"```python\n", "\n```"}},
				ExecutionResultDelimiters: &[2]string{"```tool_output\n", "\n```"},
			},
		},
		{
			name: "explicit zero retry attempts",
			in:   codeexecutor.Settings{ErrorRetryAttempts: new(0)},
			want: codeexecutor.Settings{
				ErrorRetryAttempts:        new(0),
				CodeBlockDelimiters:       [][2]string{{"```tool_code\n", "\n```"}, {"```python\n", "\n```"}},
				ExecutionResultDelimiters: &[2]string{"```tool_output\n", "\n```"},
			},
		},
		{
			name: "non-nil empty code block delimiters",
			in:   codeexecutor.Settings{CodeBlockDelimiters: [][2]string{}},
			want: codeexecutor.Settings{
				ErrorRetryAttempts:        new(2),
				CodeBlockDelimiters:       [][2]string{},
				ExecutionResultDelimiters: &[2]string{"```tool_output\n", "\n```"},
			},
		},
		{
			name: "every field set",
			in: codeexecutor.Settings{
				OptimizeDataFile:          true,
				Stateful:                  true,
				ErrorRetryAttempts:        new(-1),
				CodeBlockDelimiters:       [][2]string{{"<code>", "</code>"}},
				ExecutionResultDelimiters: &[2]string{"", ""},
				TimeoutSeconds:            new(30),
			},
			want: codeexecutor.Settings{
				OptimizeDataFile:          true,
				Stateful:                  true,
				ErrorRetryAttempts:        new(-1),
				CodeBlockDelimiters:       [][2]string{{"<code>", "</code>"}},
				ExecutionResultDelimiters: &[2]string{"", ""},
				TimeoutSeconds:            new(30),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.in.WithDefaults()
			// cmp tells a nil slice from an empty one, so the empty-delimiters
			// case fails if WithDefaults fills it in or turns it into nil.
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("WithDefaults() mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("result does not share the input's delimiters", func(t *testing.T) {
		in := codeexecutor.Settings{CodeBlockDelimiters: [][2]string{{"<code>", "</code>"}}}
		got := in.WithDefaults()
		got.CodeBlockDelimiters[0] = [2]string{"<a>", "</a>"}
		if diff := cmp.Diff([][2]string{{"<code>", "</code>"}}, in.CodeBlockDelimiters); diff != "" {
			t.Errorf("input CodeBlockDelimiters after editing the result mismatch (-want +got):\n%s", diff)
		}
	})
}
