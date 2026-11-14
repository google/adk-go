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

package tool_test

import (
	"fmt"
	"strings"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

func TestAsToolset_ListsExactlyTheTool(t *testing.T) {
	wrapped := newFunctionTool("plain_1")
	ts := tool.AsToolset(wrapped)

	if got := ts.Name(); got != "plain_1" {
		t.Errorf("Name() = %q, want %q", got, "plain_1")
	}
	for _, tc := range []struct {
		name string
		ctx  agent.ReadonlyContext
	}{
		{name: "nil context", ctx: nil},
		{name: "context", ctx: &testContext{Context: t.Context()}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ts.Tools(tc.ctx)
			if err != nil {
				t.Fatalf("Tools() error = %v, want nil", err)
			}
			if len(got) != 1 || got[0] != tool.Tool(wrapped) {
				t.Fatalf("Tools() = %v, want exactly the wrapped tool", got)
			}
			got[0] = nil
			again, err := ts.Tools(tc.ctx)
			if err != nil || len(again) != 1 || again[0] != tool.Tool(wrapped) {
				t.Errorf("Tools() after the caller changed an earlier result = %v, %v; want the wrapped tool and nil", again, err)
			}
		})
	}
}

func TestAsToolset_HasNoRequestHook(t *testing.T) {
	type requestProcessor interface {
		ProcessRequest(agent.Context, *model.LLMRequest) error
	}
	wrapped := newFunctionTool("plain_1")
	ts := tool.AsToolset(wrapped)

	if _, ok := ts.(requestProcessor); ok {
		t.Errorf("AsToolset(%q) has a ProcessRequest method; want none, so the flow runs the tool's hook once", wrapped.Name())
	}
	got, err := ts.Tools(nil)
	if err != nil || len(got) != 1 {
		t.Fatalf("Tools(nil) = %v, %v; want one tool and nil", got, err)
	}
	if _, ok := got[0].(requestProcessor); !ok {
		t.Errorf("listed tool has no ProcessRequest method; want the wrapped tool's own hook")
	}
}

func TestAsToolset_NilToolPanics(t *testing.T) {
	defer func() {
		p := recover()
		msg, ok := p.(string)
		if !ok || !strings.Contains(msg, "AsToolset") {
			t.Errorf("AsToolset(nil) panic value = %v (%T), want a string naming AsToolset", p, p)
		}
	}()
	ts := tool.AsToolset(nil)
	t.Errorf("AsToolset(nil) returned %s, want a panic", fmt.Sprint(ts))
}
