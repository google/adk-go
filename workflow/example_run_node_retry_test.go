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

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/workflow"
)

// RunNode runs a child again when an attempt fails, as the child's
// NodeConfig.RetryConfig allows, and returns the output of the attempt that
// succeeds. agent.AttemptCount tells the child which attempt it is on.
func ExampleRunNode_retry() {
	retry := workflow.DefaultRetryConfig()
	retry.InitialDelay = 0 // retry at once, for the example
	fetch := workflow.NewFunctionNode("fetch", func(ctx agent.Context, _ any) (string, error) {
		n := agent.AttemptCount(ctx)
		fmt.Println("attempt", n)
		if n == 1 {
			return "", errors.New("service unavailable")
		}
		return "ok", nil
	}, workflow.NodeConfig{RetryConfig: retry})
	caller := workflow.NewDynamicNode("caller", func(ctx agent.Context, _ any, _ func(*session.Event) error) (string, error) {
		out, err := workflow.RunNode[string](ctx, fetch, nil)
		if err != nil {
			return "", err
		}
		fmt.Println("result:", out)
		return out, nil
	}, workflow.NodeConfig{})
	w, err := workflow.New("app", workflow.Chain(workflow.Start, caller))
	if err != nil {
		log.Fatal(err)
	}

	if err := runWorkflow(w, "go"); err != nil {
		log.Fatal(err)
	}
	// Output:
	// attempt 1
	// attempt 2
	// result: ok
}
