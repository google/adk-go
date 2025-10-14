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
	"strings"
	"unicode"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/internal/agent/parentmap"
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
	// reference: adk-python src/google/adk/flows/llm_flows/instructions.py

	llmAgent := asLLMAgent(ctx.Agent())
	if llmAgent == nil {
		return nil // do nothing.
	}

	parents := parentmap.FromContext(ctx)

	rootAgent := asLLMAgent(parents.RootAgent(ctx.Agent()))
	if rootAgent == nil {
		rootAgent = llmAgent
	}

	// Append global instructions if set.
	if rootAgent != nil && rootAgent.internal().GlobalInstruction != "" {
		instructions, err := injectSessionState(ctx, llmAgent.internal().GlobalInstruction)
		if err != nil {
			return fmt.Errorf("error injecting state in global instruction: `%w`", err)
		}
		utils.AppendInstructions(req, instructions)
	}

	// Append agent's instruction
	if llmAgent.internal().Instruction != "" {
		instructions, err := injectSessionState(ctx, llmAgent.internal().Instruction)
		if err != nil {
			return fmt.Errorf("error injecting state in global instruction: `%w`", err)
		}
		utils.AppendInstructions(req, instructions)
	}

	return nil
}

// The regex to find placeholders like {variable} or {artifact.file_name}.
var placeholderRegex = regexp.MustCompile(`{+[^{}]*}+`)

// replaceMatch tries to retrieve the match from the state map
func replaceMatch(ctx agent.InvocationContext, match string) (string, error) {
	// Trim curly braces: "{var_name}" -> "var_name"
	varName := strings.TrimSpace(strings.Trim(match, "{}"))
	optional := false
	if strings.HasSuffix(varName, "?") {
		optional = true
		varName = strings.TrimSuffix(varName, "?")
	}

	if strings.HasPrefix(varName, "artifact.") {
		fileName := strings.TrimPrefix(varName, "artifact.")
		if ctx.Artifacts() == nil {
			return "", fmt.Errorf("artifact service is not initialized")
		}
		artifact, err := ctx.Artifacts().Load(fileName)
		if err != nil {
			if optional {
				return "", nil
			}
			return "", fmt.Errorf("failed to load artifact %s: %w", fileName, err)
		}
		return artifact.Text, nil
	}

	if !isValidStateName(varName) {
		return match, nil // Return the original string if not a valid name
	}

	value, err := ctx.Session().State().Get(varName)
	if err != nil {
		if optional {
			return "", nil
		}
		return "", fmt.Errorf("context variable not found: `%s`", varName)
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
		for _, p := range validPrefixes {
			if prefix == p {
				return isIdentifier(parts[1])
			}
		}
	}
	return false
}

// injectSessionState populates values in an instruction template from a context.
func injectSessionState(ctx agent.InvocationContext, template string) (string, error) {
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
