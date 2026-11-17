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

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/workflow"
)

// A node's own state schema replaces the workflow's while the node runs: the
// node may write the keys its schema declares, and a key that only the
// workflow's schema declares is refused. The next node in the graph is
// checked against the workflow's schema again.
func ExampleNodeConfig_stateSchema() {
	resolve := func(s *jsonschema.Schema) *jsonschema.Resolved {
		r, err := s.Resolve(nil)
		if err != nil {
			log.Fatal(err)
		}
		return r
	}
	workflowSchema := resolve(&jsonschema.Schema{
		Title:      "OrderSchema",
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"total": {Type: "integer"}},
	})
	paymentSchema := resolve(&jsonschema.Schema{
		Title:      "PaymentSchema",
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"card_token": {Type: "string"}},
	})

	report := func(node, key string, err error) {
		if se, ok := errors.AsType[*workflow.StateSchemaError](err); ok {
			fmt.Printf("%s: %s refused by %s (declared %v)\n", node, key, se.Schema, se.Declared)
			return
		}
		fmt.Printf("%s: %s written (error %v)\n", node, key, err)
	}
	pay := workflow.NewFunctionNode("pay", func(ctx agent.Context, _ any) (string, error) {
		report("pay", "card_token", ctx.State().Set("card_token", "tok_1"))
		report("pay", "total", ctx.State().Set("total", 42))
		return "paid", nil
	}, workflow.NodeConfig{StateSchema: paymentSchema})
	ship := workflow.NewFunctionNode("ship", func(ctx agent.Context, _ any) (string, error) {
		report("ship", "total", ctx.State().Set("total", 42))
		report("ship", "card_token", ctx.State().Set("card_token", "tok_2"))
		return "shipped", nil
	}, workflow.NodeConfig{})

	w, err := workflow.New("order", workflow.Chain(workflow.Start, pay, ship), workflow.WithStateSchema(workflowSchema))
	if err != nil {
		log.Fatal(err)
	}
	if err := runWorkflow(w, "place the order"); err != nil {
		log.Fatal(err)
	}
	// Output:
	// pay: card_token written (error <nil>)
	// pay: total refused by PaymentSchema (declared [card_token])
	// ship: total written (error <nil>)
	// ship: card_token refused by OrderSchema (declared [total])
}
