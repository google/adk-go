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

package toolregistry_test

import (
	"errors"
	"testing"

	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/adk/toolregistry"
)

type adder struct{}

func (a *adder) Name() string          { return "add" }
func (a *adder) Description() string   { return "Add two numbers" }
func (a *adder) IsLongRunning() bool   { return false }

func mustTool(t *testing.T, name string) tool.Tool {
	t.Helper()
	tt, err := functiontool.New[struct{}, string](
		functiontool.Config{Name: name, Description: name + " desc"},
		func(_ tool.Context, _ struct{}) (string, error) { return "", nil },
	)
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}
	return tt
}

func TestRegister_RejectsEmptyName(t *testing.T) {
	r := toolregistry.New()
	err := r.Register(toolregistry.Info{}, func() (tool.Tool, error) { return nil, nil })
	if err == nil {
		t.Error("expected error for empty name")
	}
}

func TestRegister_RejectsNilBuilder(t *testing.T) {
	r := toolregistry.New()
	err := r.Register(toolregistry.Info{Name: "x"}, nil)
	if err == nil {
		t.Error("expected error for nil builder")
	}
}

func TestRegisterTool_RejectsNilTool(t *testing.T) {
	r := toolregistry.New()
	err := r.RegisterTool(nil, toolregistry.Info{Name: "x"})
	if err == nil {
		t.Error("expected error for nil tool")
	}
}

func TestRegisterTool_DefaultsNameFromTool(t *testing.T) {
	r := toolregistry.New()
	tt := mustTool(t, "calc")
	if err := r.RegisterTool(tt, toolregistry.Info{Description: "calc desc"}); err != nil {
		t.Fatalf("RegisterTool: %v", err)
	}
	if !r.Has("calc") {
		t.Error("expected calc to be registered")
	}
}

func TestGet_LazilyBuildsAndCaches(t *testing.T) {
	r := toolregistry.New()
	calls := 0
	tt := mustTool(t, "lazy")
	r.Register(toolregistry.Info{Name: "lazy"}, func() (tool.Tool, error) {
		calls++
		return tt, nil
	})
	for i := 0; i < 3; i++ {
		got, err := r.Get("lazy")
		if err != nil {
			t.Fatalf("Get: %v", err)
		}
		if got != tt {
			t.Error("Get returned different instance")
		}
	}
	if calls != 1 {
		t.Errorf("builder calls = %d, want 1 (cached)", calls)
	}
}

func TestGet_PropagatesBuilderError(t *testing.T) {
	r := toolregistry.New()
	want := errors.New("build broke")
	r.Register(toolregistry.Info{Name: "broken"}, func() (tool.Tool, error) { return nil, want })
	if _, err := r.Get("broken"); !errors.Is(err, want) {
		t.Errorf("err = %v, want wraps %v", err, want)
	}
}

func TestGet_UnknownReturnsError(t *testing.T) {
	r := toolregistry.New()
	if _, err := r.Get("missing"); err == nil {
		t.Error("expected error")
	}
}

func TestList_FiltersByQuery(t *testing.T) {
	r := toolregistry.New()
	r.RegisterTool(mustTool(t, "calc"), toolregistry.Info{Name: "calc", Description: "math operations", Tags: []string{"math"}})
	r.RegisterTool(mustTool(t, "search"), toolregistry.Info{Name: "search", Description: "web search", Tags: []string{"web"}})
	r.RegisterTool(mustTool(t, "sum"), toolregistry.Info{Name: "sum", Description: "add things", Tags: []string{"math"}})

	got := r.List(toolregistry.Filter{Query: "math"})
	if len(got) != 2 {
		t.Errorf("query=math: got %d, want 2 (calc + sum match by tag)", len(got))
	}
	got = r.List(toolregistry.Filter{Tags: []string{"web"}})
	if len(got) != 1 || got[0].Name != "search" {
		t.Errorf("tags=[web]: got %v", got)
	}
	got = r.List(toolregistry.Filter{Query: "WEB"}) // case-insensitive
	if len(got) != 1 || got[0].Name != "search" {
		t.Errorf("query case-insensitive failed: got %v", got)
	}
}

func TestList_ReturnsSortedByName(t *testing.T) {
	r := toolregistry.New()
	r.RegisterTool(mustTool(t, "c_tool"), toolregistry.Info{Name: "c_tool"})
	r.RegisterTool(mustTool(t, "a_tool"), toolregistry.Info{Name: "a_tool"})
	r.RegisterTool(mustTool(t, "b_tool"), toolregistry.Info{Name: "b_tool"})
	got := r.List(toolregistry.Filter{})
	if len(got) != 3 {
		t.Fatalf("len = %d", len(got))
	}
	if got[0].Name != "a_tool" || got[1].Name != "b_tool" || got[2].Name != "c_tool" {
		t.Errorf("not sorted: %v", got)
	}
}

func TestNames(t *testing.T) {
	r := toolregistry.New()
	r.RegisterTool(mustTool(t, "y"), toolregistry.Info{Name: "y"})
	r.RegisterTool(mustTool(t, "x"), toolregistry.Info{Name: "x"})
	names := r.Names()
	if len(names) != 2 || names[0] != "x" || names[1] != "y" {
		t.Errorf("Names = %v, want [x y]", names)
	}
}
