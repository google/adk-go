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

package codeexecutor_test

import (
	"context"
	"fmt"
	"log"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/codeexecutor"
)

// echoExecutor is a minimal custom executor: instead of running the code, it
// reports the code as its standard output and exit code 0.
type echoExecutor struct{}

func (echoExecutor) Settings() codeexecutor.Settings { return codeexecutor.DefaultSettings() }

func (echoExecutor) ExecuteCode(ctx context.Context, _ agent.InvocationContext, in *codeexecutor.Input) (*codeexecutor.Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("execute code: %w", err)
	}
	exit := 0
	return &codeexecutor.Result{Stdout: in.Code, ExitCode: &exit}, nil
}

var _ codeexecutor.CodeExecutor = echoExecutor{}

// fencedBuiltIn is a built-in executor with its own code block delimiters: it
// embeds *codeexecutor.BuiltInExecutor, so it still implements
// codeexecutor.BuiltIn, and overrides only Settings.
type fencedBuiltIn struct{ *codeexecutor.BuiltInExecutor }

func (fencedBuiltIn) Settings() codeexecutor.Settings {
	return codeexecutor.Settings{CodeBlockDelimiters: [][2]string{{"<code>", "</code>"}}}
}

var _ codeexecutor.BuiltIn = fencedBuiltIn{BuiltInExecutor: codeexecutor.NewBuiltInExecutor()}

// This example calls a custom executor the way a tool does: the tool passes
// its agent.Context, which embeds agent.InvocationContext, as invocationCtx.
// A StrictContextMock stands in for the tool's context here.
func ExampleCodeExecutor() {
	ctx := context.Background()
	var toolCtx agent.Context = &agent.StrictContextMock{Ctx: ctx}

	var executor codeexecutor.CodeExecutor = echoExecutor{}
	res, err := executor.ExecuteCode(ctx, toolCtx, &codeexecutor.Input{Code: "print('hello')"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("stdout: %s\n", res.Stdout)
	fmt.Printf("stderr: %q\n", res.Stderr)
	fmt.Printf("exit code: %d\n", *res.ExitCode)
	fmt.Printf("retry attempts: %d\n", *executor.Settings().WithDefaults().ErrorRetryAttempts)

	// Output:
	// stdout: print('hello')
	// stderr: ""
	// exit code: 0
	// retry attempts: 2
}
