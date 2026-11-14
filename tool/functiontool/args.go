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

package functiontool

import (
	"fmt"
	"maps"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/internal/typeutil"
)

// dropUndeclared returns args without the keys s does not declare, when s
// forbids extra properties, as the schema jsonschema-go infers for every
// struct does. adk-python drops an argument the function does not declare
// (tools/function_tool.py:284). A schema that allows extra properties, such as
// a map's, keeps every key, because its handler asked for them. args is not
// modified; a copy is returned when a key is dropped.
func dropUndeclared(s *jsonschema.Schema, args map[string]any) map[string]any {
	if s == nil || !typeutil.IsFalseSchema(s.AdditionalProperties) {
		return args
	}
	var out map[string]any
	for k := range args {
		if _, ok := s.Properties[k]; ok {
			continue
		}
		if out == nil {
			out = maps.Clone(args)
		}
		delete(out, k)
	}
	if out == nil {
		return args
	}
	return out
}

// missingMandatory returns the names in s's required list, in that list's
// order, that args does not hold. For an inferred struct schema that is field
// order, and a field with omitempty or omitzero is not mandatory.
func missingMandatory(s *jsonschema.Schema, args map[string]any) []string {
	if s == nil {
		return nil
	}
	var missing []string
	for _, name := range s.Required {
		if _, ok := args[name]; !ok {
			missing = append(missing, name)
		}
	}
	return missing
}

// missingArgsText is adk-python's text for a call without all mandatory
// arguments (tools/function_tool.py:368-371), which the model reads to retry.
func missingArgsText(toolName string, missing []string) string {
	return fmt.Sprintf("Invoking `%s()` failed as the following mandatory input parameters are not present:\n%s\n"+
		"You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters.",
		toolName, strings.Join(missing, "\n"))
}
