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

package agent

import (
	"context"
	"testing"
)

// fakeToolContext shows the intended usage: embed StrictContextMock and
// override only the methods the test needs. This mirrors how out-of-tree
// consumers build a ToolContext test double without tracking every method.
type fakeToolContext struct {
	StrictContextMock
}

var _ ToolContext = (*fakeToolContext)(nil)

func TestStrictContextMock_ValueDelegatesToCtx(t *testing.T) {
	type key struct{}
	ctx := context.WithValue(context.Background(), key{}, "v")

	f := &fakeToolContext{StrictContextMock{Ctx: ctx}}

	if got := f.Value(key{}); got != "v" {
		t.Errorf("Value() = %v, want %q", got, "v")
	}
}

func TestStrictContextMock_ADKMethodPanics(t *testing.T) {
	f := &fakeToolContext{StrictContextMock{Ctx: context.Background()}}

	defer func() {
		if r := recover(); r == nil {
			t.Error("FunctionCallID() did not panic, want panic for unimplemented method")
		}
	}()
	_ = f.FunctionCallID()
}

func TestStrictContextMock_NilCtxPanics(t *testing.T) {
	f := &fakeToolContext{}

	defer func() {
		if r := recover(); r == nil {
			t.Error("Value() did not panic with nil Ctx")
		}
	}()
	_ = f.Value("k")
}
