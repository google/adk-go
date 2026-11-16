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

// Check Err after the last builder call and before workflow.New: an empty
// routing map adds no edge, so the edges Build returns would otherwise
// describe a graph without the routes the caller meant to add.
func ExampleEdgeBuilder_Err() {
	classify := workflow.NewFunctionNode("classify", func(_ agent.Context, in string) (string, error) {
		return in, nil
	}, workflow.NodeConfig{})

	b := workflow.NewEdgeBuilder().
		Add(workflow.Start, classify).
		AddRoutes(classify, nil)

	fmt.Println(errors.Is(b.Err(), workflow.ErrEmptyRoutingMap))
	fmt.Println(len(b.Build()))
	// Output:
	// true
	// 1
}
