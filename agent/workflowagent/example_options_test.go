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

package workflowagent_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// A workflow agent takes the state schema and the concurrency cap of
// workflow.New as Config fields. Here the cap of 1 runs the three branches
// one at a time, in the order they were triggered, and the state schema
// refuses the last branch's write of a key it does not declare.
func ExampleNew_stateSchemaAndMaxConcurrency() {
	schema, err := (&jsonschema.Schema{
		Title:      "Report",
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"pages": {Type: "integer"}},
	}).Resolve(nil)
	if err != nil {
		log.Fatal(err)
	}

	var mu sync.Mutex
	var started []string
	branch := func(name, key string) workflow.Node {
		return workflow.NewFunctionNode(name, func(ctx agent.Context, _ any) (string, error) {
			mu.Lock()
			started = append(started, name)
			mu.Unlock()
			return name, ctx.State().Set(key, 1)
		}, workflow.NodeConfig{})
	}
	a, b, c := branch("a", "pages"), branch("b", "pages"), branch("c", "author")
	edges := workflow.NewEdgeBuilder().
		AddFanOut(workflow.Start, a, b, c).
		AddFanIn(workflow.NewJoinNode("join"), a, b, c).
		Build()

	wa, err := workflowagent.New(workflowagent.Config{
		Name:           "report",
		Edges:          edges,
		StateSchema:    schema,
		MaxConcurrency: 1,
	})
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		log.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: wa, SessionService: sessions})
	if err != nil {
		log.Fatal(err)
	}
	for _, err := range r.Run(ctx, "user", created.Session.ID(), genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if se, ok := errors.AsType[*workflow.StateSchemaError](err); ok {
			fmt.Printf("refused: key %q, schema %q\n", se.Key, se.Schema)
		}
	}
	fmt.Println(started)
	// Output:
	// refused: key "author", schema "Report"
	// [a b c]
}
