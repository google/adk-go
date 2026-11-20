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
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/workflow"
)

// A node whose run takes longer than its NodeConfig.Timeout fails the run
// with a *NodeTimeoutError that names the node and its timeout. The error
// still matches context.DeadlineExceeded with errors.Is.
func ExampleNodeTimeoutError() {
	slow := workflow.NewFunctionNode("slow", func(ctx agent.Context, _ any) (string, error) {
		<-ctx.Done()
		return "", ctx.Err()
	}, workflow.NodeConfig{Timeout: 50 * time.Millisecond})
	w, err := workflow.New("fetch", workflow.Chain(workflow.Start, slow))
	if err != nil {
		log.Fatal(err)
	}

	err = runWorkflow(w, "go")

	if timeout, ok := errors.AsType[*workflow.NodeTimeoutError](err); ok {
		fmt.Println(timeout)
	}
	// Output:
	// node "slow" timed out after 0.05 seconds
}
