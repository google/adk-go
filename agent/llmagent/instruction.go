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

package llmagent

import (
	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/llminternal"
)

// InjectSessionState populates values in the instruction template, e.g. state,
// artifact, etc.
//
// {var} is used to insert the value of the state variable named var.
// {artifact.var} is used to insert the text content of the artifact named var.
// State key identifier should be match "^[a-zA-Z_][a-zA-Z0-9_]*$".
// Otherwise it will be treated as a literal.
//
// If the state variable or artifact does not exist, the agent will raise an
// error. If you want to ignore the error, you can append a ? to the
// variable name as in {var?}.
//
// This method is intended to be used in InstructionProvider based Instruction
// and GlobalInstruction which are called with ReadonlyContext.
func InjectSessionState(ctx agent.InvocationContext, template string) (string, error) {
	return llminternal.InjectSessionState(ctx, template)
}
