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

package toolutils_test

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/tool/toolutils"
)

// fakeTool is a minimal implementation of toolutils.Packable for tests.
type fakeTool struct {
	name string
	decl *genai.FunctionDeclaration
}

func (f fakeTool) Name() string                            { return f.name }
func (f fakeTool) Declaration() *genai.FunctionDeclaration { return f.decl }

var _ toolutils.Packable = fakeTool{}

func TestPackTool_SingleTool(t *testing.T) {
	req := &model.LLMRequest{}
	tool := fakeTool{name: "first", decl: &genai.FunctionDeclaration{Name: "first"}}

	if err := toolutils.PackTool(req, tool); err != nil {
		t.Fatalf("PackTool() returned error: %v", err)
	}

	if _, ok := req.Tools["first"]; !ok {
		t.Fatalf("req.Tools missing %q; got %v", "first", req.Tools)
	}
	if got := len(req.Config.Tools); got != 1 {
		t.Fatalf("len(req.Config.Tools) = %d, want 1", got)
	}
	if got := len(req.Config.Tools[0].FunctionDeclarations); got != 1 {
		t.Fatalf("len(FunctionDeclarations) = %d, want 1", got)
	}
}

func TestPackTool_SecondToolSharesSameGenaiTool(t *testing.T) {
	req := &model.LLMRequest{}
	first := fakeTool{name: "first", decl: &genai.FunctionDeclaration{Name: "first"}}
	second := fakeTool{name: "second", decl: &genai.FunctionDeclaration{Name: "second"}}

	if err := toolutils.PackTool(req, first); err != nil {
		t.Fatalf("PackTool(first) returned error: %v", err)
	}
	if err := toolutils.PackTool(req, second); err != nil {
		t.Fatalf("PackTool(second) returned error: %v", err)
	}

	if _, ok := req.Tools["second"]; !ok {
		t.Fatalf("req.Tools missing %q; got %v", "second", req.Tools)
	}
	// Both declarations must live on a single genai.Tool, not two.
	if got := len(req.Config.Tools); got != 1 {
		t.Fatalf("len(req.Config.Tools) = %d, want 1 (second tool should append, not create)", got)
	}
	if got := len(req.Config.Tools[0].FunctionDeclarations); got != 2 {
		t.Fatalf("len(FunctionDeclarations) = %d, want 2", got)
	}
}

func TestPackTool_DuplicateName(t *testing.T) {
	req := &model.LLMRequest{}
	tool := fakeTool{name: "dup", decl: &genai.FunctionDeclaration{Name: "dup"}}

	if err := toolutils.PackTool(req, tool); err != nil {
		t.Fatalf("first PackTool() returned error: %v", err)
	}
	if err := toolutils.PackTool(req, tool); err == nil {
		t.Fatalf("second PackTool() with duplicate name = nil error, want error")
	}
}

func TestPackTool_NilDeclaration(t *testing.T) {
	req := &model.LLMRequest{}
	tool := fakeTool{name: "nodecl", decl: nil}

	if err := toolutils.PackTool(req, tool); err != nil {
		t.Fatalf("PackTool() returned error: %v", err)
	}

	if _, ok := req.Tools["nodecl"]; !ok {
		t.Fatalf("req.Tools missing %q; nil-declaration tool should still register", "nodecl")
	}
	if got := len(req.Config.Tools); got != 0 {
		t.Fatalf("len(req.Config.Tools) = %d, want 0 (nil declaration adds nothing)", got)
	}
}
