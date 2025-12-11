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

package llminternal

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
	"unicode"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/parentmap"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
)

// TODO: Remove this once state keywords are implemented and replace with those consts
const (
	appPrefix  = "app:"
	userPrefix = "user:"
	tempPrefix = "temp:"
)

// instructionsRequestProcessor configures req's instructions and global instructions for LLM flow.
func instructionsRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest) error {
	llmAgent := asLLMAgent(ctx.Agent())
	if llmAgent == nil {
		return nil // do nothing.
	}

	parents := parentmap.FromContext(ctx)

	rootAgent := asLLMAgent(parents.RootAgent(ctx.Agent()))
	if rootAgent == nil {
		rootAgent = llmAgent
	}

	// Append global instructions.
	if err := appendGlobalInstructions(ctx, req, rootAgent.internal()); err != nil {
		return fmt.Errorf("failed to append global instructions: %w", err)
	}

	// Append agent's instruction
	if err := appendInstructions(ctx, req, llmAgent.internal()); err != nil {
		return fmt.Errorf("failed to append instructions: %w", err)
	}

	return nil
}

// The regex to find placeholders like {variable} or {artifact.file_name}.
var placeholderRegex = regexp.MustCompile(`{+[^{}]*}+`)

func appendInstructions(ctx agent.InvocationContext, req *model.LLMRequest, agentState *State) error {
	if agentState.InstructionProvider != nil {
		instruction, err := agentState.InstructionProvider(icontext.NewReadonlyContext(ctx))
		if err != nil {
			return fmt.Errorf("failed to evaluate global instruction provider: %w", err)
		}

		utils.AppendInstructions(req, instruction)
		return nil
	}

	if agentState.Instruction == "" {
		return nil
	}

	inst, err := InjectSessionState(ctx, agentState.Instruction)
	if err != nil {
		return fmt.Errorf("failed to inject session state into instruction: %w", err)
	}

	utils.AppendInstructions(req, inst)
	return nil
}

func appendGlobalInstructions(ctx agent.InvocationContext, req *model.LLMRequest, agentState *State) error {
	if agentState.GlobalInstructionProvider != nil {
		instruction, err := agentState.GlobalInstructionProvider(icontext.NewReadonlyContext(ctx))
		if err != nil {
			return fmt.Errorf("failed to evaluate global instruction provider: %w", err)
		}

		utils.AppendInstructions(req, instruction)
		return nil
	}

	if agentState.GlobalInstruction == "" {
		return nil
	}

	inst, err := InjectSessionState(ctx, agentState.GlobalInstruction)
	if err != nil {
		return fmt.Errorf("failed to inject session state into global instruction: %w", err)
	}

	utils.AppendInstructions(req, inst)
	return nil
}

// replaceMatch is the Go equivalent of the _replace_match async function in the Python code.
func replaceMatch(ctx agent.InvocationContext, match string) (string, error) {
	// Trim curly braces: "{var_name}" -> "var_name"
	varName := strings.TrimSpace(strings.Trim(match, "{}"))
	optional := false
	if strings.HasSuffix(varName, "?") {
		optional = true
		varName = strings.TrimSuffix(varName, "?")
	}

	if after, ok := strings.CutPrefix(varName, "artifact."); ok {
		fileName := after
		if ctx.Artifacts() == nil {
			return "", fmt.Errorf("artifact service is not initialized")
		}
		resp, err := ctx.Artifacts().Load(ctx, fileName)
		if err != nil {
			if optional {
				// TODO: consistent logging approach in adk-go
				return "", nil
			}
			return "", fmt.Errorf("failed to load artifact %s: %w", fileName, err)
		}
		return resp.Part.Text, nil
	}

	if !isValidStateName(varName) {
		return match, nil // Return the original string if not a valid name
	}

	value, err := ctx.Session().State().Get(varName)
	if err != nil {
		if optional {
			// TODO: log error when !errors.Is(err, session.ErrStateKeyNotExist)
			return "", nil
		}
		return "", err
	}

	if value == nil {
		return "", nil
	}

	return fmt.Sprintf("%v", value), nil
}

// isIdentifier checks if a string is a valid Go identifier.
// This is the equivalent of Python's `str.isidentifier()`.
func isIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if i == 0 {
			if !unicode.IsLetter(r) && r != '_' {
				return false
			}
		} else {
			if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' {
				return false
			}
		}
	}
	return true
}

// isValidStateName checks if the variable name is a valid state name.
func isValidStateName(varName string) bool {
	parts := strings.Split(varName, ":")
	if len(parts) == 1 {
		return isIdentifier(varName)
	}

	if len(parts) == 2 {
		prefix := parts[0] + ":"
		validPrefixes := []string{appPrefix, userPrefix, tempPrefix}
		if slices.Contains(validPrefixes, prefix) {
			return isIdentifier(parts[1])
		}
	}
	return false
}

// InjectSessionState populates values in an instruction template from a context.
func InjectSessionState(ctx agent.InvocationContext, template string) (string, error) {
	// Find all matches, then iterate through them, building the result string.
	var result strings.Builder
	lastIndex := 0
	matches := placeholderRegex.FindAllStringIndex(template, -1)

	for _, matchIndexes := range matches {
		startIndex, endIndex := matchIndexes[0], matchIndexes[1]

		// Append the text between the last match and this one
		result.WriteString(template[lastIndex:startIndex])

		// Get the replacement for the current match
		matchStr := template[startIndex:endIndex]
		replacement, err := replaceMatch(ctx, matchStr)
		if err != nil {
			return "", err // Propagate the error
		}
		result.WriteString(replacement)

		lastIndex = endIndex
	}

	// Append any remaining text after the last match
	result.WriteString(template[lastIndex:])

	return result.String(), nil
}
