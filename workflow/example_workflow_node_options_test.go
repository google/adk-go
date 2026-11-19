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
	"fmt"
	"log"
	"sync"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/workflow"
)

// A nested workflow takes the options of New. Here its cap of 1 runs its
// three parallel branches one at a time, in the order they were triggered,
// although the enclosing workflow has no cap.
func ExampleNewWorkflowNodeWithOptions() {
	var mu sync.Mutex
	var started []string
	branch := func(name string) workflow.Node {
		return workflow.NewFunctionNode(name, func(agent.Context, any) (string, error) {
			mu.Lock()
			defer mu.Unlock()
			started = append(started, name)
			return name, nil
		}, workflow.NodeConfig{})
	}
	a, b, c := branch("a"), branch("b"), branch("c")
	edges := workflow.NewEdgeBuilder().
		AddFanOut(workflow.Start, a, b, c).
		AddFanIn(workflow.NewJoinNode("join"), a, b, c).
		Build()

	research, err := workflow.NewWorkflowNodeWithOptions("research", edges, workflow.WithMaxConcurrency(1))
	if err != nil {
		log.Fatal(err)
	}
	outer, err := workflow.New("outer", workflow.Chain(workflow.Start, research))
	if err != nil {
		log.Fatal(err)
	}
	if err := runWorkflow(outer, "go"); err != nil {
		log.Fatal(err)
	}
	fmt.Println(started)
	// Output:
	// [a b c]
}
