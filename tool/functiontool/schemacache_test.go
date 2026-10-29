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

package functiontool_test

import (
	"iter"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/schemacache"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// TestNew_RemembersSchemas checks that the schemas a function tool declares
// are remembered, which is what lets a model adapter send them without
// encoding them on every call.
func TestNew_RemembersSchemas(t *testing.T) {
	sum, err := functiontool.New(functiontool.Config{Name: "sum", Description: "sums"}, sumFunc)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := functiontool.NewStreaming(functiontool.Config{Name: "stream", Description: "streams"},
		func(agent.Context, SumArgs) iter.Seq2[string, error] { return func(func(string, error) bool) {} })
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		schema any
	}{
		{"New parameters", sum.(toolutils.Tool).Declaration().ParametersJsonSchema},
		{"New response", sum.(toolutils.Tool).Declaration().ResponseJsonSchema},
		{"NewStreaming parameters", stream.(toolutils.Tool).Declaration().ParametersJsonSchema},
	} {
		if _, ok := schemacache.Encoded(tc.schema); !ok {
			t.Errorf("%s schema (%T) is not remembered", tc.name, tc.schema)
		}
	}
}
