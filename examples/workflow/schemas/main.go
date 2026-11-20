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

// Command schemas shows a root workflow with an input schema and an
// output schema. It runs three user messages through a runner and prints
// what the workflow's node received and how each run ended:
//
//   - a valid message, which reaches the node with a default filled in;
//   - a message that fails the input schema, which stops the run before
//     any node runs;
//   - a message whose terminal output fails the output schema, which ends
//     the run with an error after the node has run.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// request is the workflow's input. units is optional and defaults to
// "metric".
var inputSchema = &jsonschema.Schema{
	Type:     "object",
	Required: []string{"city"},
	Properties: map[string]*jsonschema.Schema{
		"city":  {Type: "string"},
		"units": {Type: "string", Enum: []any{"metric", "imperial"}, Default: []byte(`"metric"`)},
	},
}

// forecast is the workflow's output. source is required but has a
// default, so a node may leave it out.
var outputSchema = &jsonschema.Schema{
	Type:     "object",
	Required: []string{"city", "summary", "source"},
	Properties: map[string]*jsonschema.Schema{
		"city":    {Type: "string"},
		"summary": {Type: "string", MinLength: new(1)},
		"source":  {Type: "string", Default: []byte(`"example"`)},
	},
}

// forecast receives the validated input as a map, with the default of
// units already filled in. It writes no summary for an unknown city, so
// that run fails the output schema.
func forecast(_ agent.Context, in map[string]any) (map[string]any, error) {
	fmt.Printf("  node input: %v\n", in)
	summaries := map[string]string{"Paris": "sunny, 21 degrees", "Oslo": "rain, 9 degrees"}
	return map[string]any{"city": in["city"], "summary": summaries[in["city"].(string)]}, nil
}

func main() {
	ctx := context.Background()

	in, err := inputSchema.Resolve(nil)
	if err != nil {
		log.Fatalf("input schema: %v", err)
	}
	out, err := outputSchema.Resolve(nil)
	if err != nil {
		log.Fatalf("output schema: %v", err)
	}

	node := workflow.NewFunctionNode("forecast", forecast, workflow.NodeConfig{})
	root, err := workflowagent.New(workflowagent.Config{
		Name:         "weather",
		Description:  "Looks up a forecast for a city",
		Edges:        workflow.Chain(workflow.Start, node),
		InputSchema:  in,
		OutputSchema: out,
	})
	if err != nil {
		log.Fatalf("workflowagent.New: %v", err)
	}

	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "schemas", Agent: root, SessionService: sessions})
	if err != nil {
		log.Fatalf("runner.New: %v", err)
	}

	for _, message := range []string{
		`{"city": "Paris"}`,
		`{"town": "Paris"}`,
		`{"city": "Atlantis", "units": "imperial"}`,
	} {
		// One session per message, so each run starts fresh.
		created, err := sessions.Create(ctx, &session.CreateRequest{AppName: "schemas", UserID: "user"})
		if err != nil {
			log.Fatalf("create session: %v", err)
		}
		fmt.Printf("message %s\n", message)
		var runErr error
		for _, err := range r.Run(ctx, "user", created.Session.ID(), genai.NewContentFromText(message, genai.RoleUser), agent.RunConfig{}) {
			if err != nil && runErr == nil {
				runErr = err
			}
		}
		switch {
		case runErr == nil:
			fmt.Println("  run finished")
		case errors.Is(runErr, workflow.ErrInputValidation):
			fmt.Printf("  input rejected: %v\n", runErr)
		case errors.Is(runErr, workflow.ErrNodeFailed):
			fmt.Printf("  output rejected: %v\n", runErr)
		default:
			log.Fatalf("run: %v", runErr)
		}
	}
}
