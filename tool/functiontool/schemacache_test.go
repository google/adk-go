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
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/schemacache"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolutils"
)

// TestNew_SchemaChangedBetweenToolsIsRecordedAgain checks that a schema
// changed after a failed New, and before the New that succeeds, is recorded
// as the successful tool declares it.
func TestNew_SchemaChangedBetweenToolsIsRecordedAgain(t *testing.T) {
	s := &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{"city": {Type: "string", Description: "old"}}}
	handler := func(agent.Context, map[string]any) (map[string]any, error) { return nil, nil }
	if _, err := functiontool.New(functiontool.Config{Name: "w", InputSchema: s, RequireConfirmationProvider: func(int) bool { return true }}, handler); err == nil {
		t.Fatal("New accepted a RequireConfirmationProvider of the wrong type")
	}
	s.Properties["city"].Description = "new"
	tl, err := functiontool.New(functiontool.Config{Name: "w", InputSchema: s}, handler)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := schemacache.Encoded(tl.(toolutils.Tool).Declaration().ParametersJsonSchema)
	if !ok || !strings.Contains(string(got), `"new"`) {
		t.Errorf("recorded encoding = %s, %v, want the description \"new\"", got, ok)
	}
}
