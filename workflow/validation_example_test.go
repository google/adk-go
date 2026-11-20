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

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/workflow"
)

// New refuses edges that do not describe a valid graph. Every such error
// matches ErrGraphValidation, and its message says which rule failed.
func ExampleErrGraphValidation() {
	echo := func(_ agent.Context, in any) (any, error) { return in, nil }
	a := workflow.NewFunctionNode("A", echo, workflow.NodeConfig{})
	b := workflow.NewFunctionNode("B", echo, workflow.NodeConfig{})

	// B leads back to A on an edge without a route, so the two would run
	// each other forever.
	_, err := workflow.New("loop", workflow.Chain(workflow.Start, a, b, a))

	fmt.Println(errors.Is(err, workflow.ErrGraphValidation))
	fmt.Println(err)
	// Output:
	// true
	// graph validation failed: unconditional cycle detected: "A" -> "B" -> "A"
}
