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

package workflow_test

import (
	"errors"
	"fmt"
	"log"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/workflow"
)

// An update is checked as a whole before anything is written, so a refused
// update leaves the state as it was. SetDefault on a key that is present
// returns the stored value and does not check its default.
func ExampleSchemaState() {
	schema, err := (&jsonschema.Schema{
		Title: "PipelineSchema",
		Type:  "object",
		Properties: map[string]*jsonschema.Schema{
			"counter": {Type: "integer"},
			"name":    {Type: "string"},
		},
	}).Resolve(nil)
	if err != nil {
		log.Fatal(err)
	}
	st, err := workflow.NewSchemaState(workflow.SchemaStateConfig{
		Schema: schema,
		Value:  map[string]any{"counter": 1},
	})
	if err != nil {
		log.Fatal(err)
	}

	// name is valid, but count is not declared, so neither is written.
	err = st.Update(map[string]any{"name": "abc", "count": 2})
	if se, ok := errors.AsType[*workflow.StateSchemaError](err); ok {
		fmt.Printf("update refused: %q is not declared in %s %v\n", se.Key, se.Schema, se.Declared)
	}
	fmt.Println("pending:", st.Delta())

	v, err := st.SetDefault("counter", "not a number")
	fmt.Println("counter:", v, err)

	// Output:
	// update refused: "count" is not declared in PipelineSchema [counter name]
	// pending: map[]
	// counter: 1 <nil>
}
