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
	"errors"
	"fmt"
	"iter"
	"log"
	"strings"
	"unicode"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// instructionsRequestProcessor configures req's instructions and global instructions for LLM flow.
func instructionsRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		llmAgent := asLLMAgent(ctx.Agent())
		if llmAgent == nil {
			return // do nothing.
		}

		parents := parentmap.FromContext(ctx)

		rootAgent := asLLMAgent(parents.RootAgent(ctx.Agent()))
		if rootAgent == nil {
			rootAgent = llmAgent
		}

		// Append global instructions.
		if err := appendGlobalInstructions(ctx, req, rootAgent.internal()); err != nil {
			yield(nil, fmt.Errorf("failed to append global instructions: %w", err))
			return
		}

		// Append agent's instruction
		if err := appendInstructions(ctx, req, llmAgent.internal()); err != nil {
			yield(nil, fmt.Errorf("failed to append instructions: %w", err))
			return
		}
	}
}

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
				log.Printf("failed to load optional artifact %s: %v", fileName, err)
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
			if !errors.Is(err, session.ErrStateKeyNotExist) {
				log.Printf("failed to get optional state key %s: %v", varName, err)
			}
			return "", nil
		}
		return "", err
	}

	if value == nil {
		return "", nil
	}

	// Fast-path string values to avoid fmt.Sprintf reflection and allocations.
	switch v := value.(type) {
	case string:
		return v, nil
	default:
		return fmt.Sprintf("%v", v), nil
	}
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
// Optimized with zero allocations by using strings.Cut and a static switch
// instead of strings.Split and slice literals.
func isValidStateName(varName string) bool {
	prefix, identifier, found := strings.Cut(varName, ":")
	if !found {
		return isIdentifier(varName)
	}
	if strings.ContainsRune(identifier, ':') {
		return false
	}
	switch prefix {
	case "app", "user", "temp":
		return isIdentifier(identifier)
	default:
		return false
	}
}

// InjectSessionState populates values in an instruction template from a context.
// Performance-optimized by Bolt: uses direct string scanning instead of regexp
// matching to eliminate heap slice allocations for regex matches.
func InjectSessionState(ctx agent.InvocationContext, template string) (string, error) {
	firstOpen := strings.IndexByte(template, '{')
	if firstOpen < 0 {
		return template, nil
	}

	var result strings.Builder
	result.Grow(len(template))

	lastIndex := 0
	i := firstOpen

	for i < len(template) {
		openRel := strings.IndexByte(template[i:], '{')
		if openRel < 0 {
			break
		}
		openIdx := i + openRel

		// Count consecutive '{'
		endOpen := openIdx + 1
		for endOpen < len(template) && template[endOpen] == '{' {
			endOpen++
		}

		// Find non-'{}' content
		idx := endOpen
		for idx < len(template) && template[idx] != '{' && template[idx] != '}' {
			idx++
		}

		if idx >= len(template) || template[idx] == '{' {
			// No closing '}' found for this '{', move to idx
			i = idx
			continue
		}

		// Found '}', collect all consecutive '}'
		closeEnd := idx + 1
		for closeEnd < len(template) && template[closeEnd] == '}' {
			closeEnd++
		}

		// Valid placeholder match: template[openIdx:closeEnd]
		result.WriteString(template[lastIndex:openIdx])

		matchStr := template[openIdx:closeEnd]
		replacement, err := replaceMatch(ctx, matchStr)
		if err != nil {
			return "", err
		}
		result.WriteString(replacement)

		lastIndex = closeEnd
		i = closeEnd
	}

	if lastIndex == 0 {
		// No valid placeholders were matched
		return template, nil
	}

	result.WriteString(template[lastIndex:])
	return result.String(), nil
}
