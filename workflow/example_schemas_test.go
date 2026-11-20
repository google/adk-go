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
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// runWorkflow runs w as the root agent of a runner, in a new in-memory
// session with one user message, and returns the run's first error.
func runWorkflow(w *workflow.Workflow, message string) error {
	ctx := context.Background()
	a, err := agent.New(agent.Config{Name: w.Name(), Run: w.Run})
	if err != nil {
		return err
	}
	sessions := session.InMemoryService()
	created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		return err
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: sessions})
	if err != nil {
		return err
	}
	msg := genai.NewContentFromText(message, genai.RoleUser)
	for _, err := range r.Run(ctx, "user", created.Session.ID(), msg, agent.RunConfig{}) {
		if err != nil {
			return err
		}
	}
	return nil
}

// A root workflow's output schema validates the output of its single
// terminal node when the run finishes. Defaults are filled before the
// check, so a required property that declares a default may be missing
// from the output. An output that still fails the schema ends the run
// with an error that wraps ErrNodeFailed.
func ExampleWithOutputSchema() {
	schema, err := (&jsonschema.Schema{
		Type:     "object",
		Required: []string{"answer", "source"},
		Properties: map[string]*jsonschema.Schema{
			"answer": {Type: "string"},
			"source": {Type: "string", Default: []byte(`"model"`)},
		},
	}).Resolve(nil)
	if err != nil {
		log.Fatal(err)
	}

	for _, output := range []map[string]any{
		{"answer": "42"},  // source is filled with its default
		{"source": "web"}, // answer is missing and has no default
	} {
		answer := workflow.NewFunctionNode("answer", func(agent.Context, any) (map[string]any, error) {
			return output, nil
		}, workflow.NodeConfig{})
		w, err := workflow.New("qa", workflow.Chain(workflow.Start, answer), workflow.WithOutputSchema(schema))
		if err != nil {
			log.Fatal(err)
		}

		err = runWorkflow(w, "What is the answer?")
		fmt.Printf("output %v: failed=%t, ErrNodeFailed=%t\n", output, err != nil, errors.Is(err, workflow.ErrNodeFailed))
	}
	// Output:
	// output map[answer:42]: failed=false, ErrNodeFailed=false
	// output map[source:web]: failed=true, ErrNodeFailed=true
}
