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

package instructionutil

import (
	"fmt"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
)

// InjectSessionState populates values in the instruction template, e.g. state,
// artifact, etc.
//   - There can be placeholders like {key_name} that will be resolved by ADK
//     at runtime using session state and context.
//   - key_name must match "^[a-zA-Z_][a-zA-Z0-9_]*$", otherwise it will be
//     treated as a literal.
//   - {artifact.key_name} can be used to insert the text content of the
//     artifact named key_name.
//
// If the state variable or artifact does not exist, the agent will raise an
// error. If you want to ignore the error, you can append a ? to the
// variable name as in {var?}.
//
// This method is intended to be used in InstructionProvider based Instruction
// and GlobalInstruction which are called with ReadonlyContext.
func InjectSessionState(ctx agent.ReadonlyContext, template string) (string, error) {
	ictx, ok := ctx.(*icontext.ReadonlyContext)
	if !ok {
		return "", fmt.Errorf("unexpected context type: %T", ctx)
	}
	return llminternal.InjectSessionState(ictx.InvocationContext, template)
}
