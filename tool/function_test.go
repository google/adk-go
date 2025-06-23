// Copyright 2025 Google LLC
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

package tool

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/google/adk-go"
	"github.com/google/go-cmp/cmp"
)

func TestFunctionTool(t *testing.T) {
	ctx := t.Context()

	for _, tc := range []struct {
		tool    *FunctionTool[testFnIn, testFnOut]
		in      testFnIn
		wantOut testFnOut
		wantErr bool
	}{
		{
			tool:    NewFunctionTool("sum", "", sumFn),
			in:      testFnIn{A: 1, B: 2},
			wantOut: testFnOut{Result: 3},
		},
		{
			tool:    NewFunctionTool("error", "", errorFn),
			in:      testFnIn{A: 1, B: 2},
			wantErr: true,
		},
	} {
		t.Run(tc.tool.Name(), func(t *testing.T) {
			toolContext := &adk.ToolContext{}

			res, err := tc.tool.Run(ctx, toolContext, toMap(stringify(tc.in)))
			if tc.wantErr && err == nil {
				t.Fatalf("tool(%v).Run=(%v, nil), want (_, <error>)", tc.tool.Name(), res)
			}
			if !tc.wantErr && (err != nil || !cmp.Equal(stringify(res), stringify(tc.wantOut))) {
				t.Fatalf("tool(%v).Run=(%v, %v), want (%v, nil)", tc.tool.Name(), res, err, tc.wantOut)
			}
		})
	}
}

func stringify(x any) json.RawMessage {
	str, err := json.Marshal(x)
	if err != nil {
		panic(err)
	}
	return str
}

func toMap(msg json.RawMessage) map[string]any {
	rawArgs, err := json.Marshal(msg)
	if err != nil {
		panic(err)
	}
	var m map[string]any
	if err := json.Unmarshal(rawArgs, &m); err != nil {
		panic(err)
	}
	return m
}

type testFnIn struct {
	A int `json:"a,omitempty"`
	B int `json:"b,omitempty"`
}

type testFnOut struct {
	Result int `json:"res,omitempty"`
}

func sumFn(ctx context.Context, _ *adk.ToolContext, in testFnIn) (testFnOut, error) {
	return testFnOut{Result: in.A + in.B}, nil
}

func errorFn(ctx context.Context, _ *adk.ToolContext, _ testFnIn) (testFnOut, error) {
	return testFnOut{}, fmt.Errorf("err")
}
