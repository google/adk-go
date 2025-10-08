// Copyright 2025 Google LLC
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

package exitlooptool

import (
	"fmt"

	"google.golang.org/adk/tool"
)

type EmptyArgs struct{}

func exitLoop(ctx tool.Context, myArgs EmptyArgs) map[string]string {
	ctx.Actions().Escalate = true
	ctx.Actions().SkipSummarization = true
	return map[string]string{}
}

func New() (tool.Tool, error) {
	exitLoopTool, err := tool.NewFunctionTool(tool.FunctionToolConfig{
		Name:        "exit_loop",
		Description: "Exits the loop.\nCall this function only when you are instructed to do so.\n",
	}, exitLoop)
	if err != nil {
		return nil, fmt.Errorf("error creating exit loop tool: %w", err)
	}
	return exitLoopTool, nil
}
