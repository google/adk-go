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
	"io/fs"
	"iter"
	"path"
	"regexp"
	"slices"
	"strings"
	"unicode"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// TODO: Remove this once state keywords are implemented and replace with those consts
const (
	appPrefix  = "app:"
	userPrefix = "user:"
	tempPrefix = "temp:"
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

// The regex to find placeholders like {variable} or {artifact.file_name}.
var placeholderRegex = regexp.MustCompile(`{+[^{}]*}+`)

// A placeholder directly after one of these bytes is literal text, e.g.
// "${expression}" or "\{expression}". adk-python expresses this as the
// negative lookbehind (?<![\$\{\\]), which RE2 does not support. The { from
// that lookbehind is not needed here: {+ always starts a match at the first
// brace of a run, so a match is never preceded by {.
const literalPlaceholderPrefixes = `$\`

func appendInstructions(ctx agent.InvocationContext, req *model.LLMRequest, agentState *State) error {
	if agentState.InstructionProvider != nil {
		instruction, err := agentState.InstructionProvider(icontext.NewReadonlyContext(ctx))
		if err != nil {
			return fmt.Errorf("failed to evaluate instruction provider: %w", err)
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
			if optional && errors.Is(err, fs.ErrNotExist) {
				return "", nil
			}
			return "", fmt.Errorf("failed to load artifact %s: %w", fileName, err)
		}
		return artifactText(fileName, resp.Part), nil
	}

	if !isValidStateName(varName) {
		return match, nil // Return the original string if not a valid name
	}

	value, err := ctx.Session().State().Get(varName)
	if err != nil {
		if optional && errors.Is(err, session.ErrStateKeyNotExist) {
			return "", nil
		}
		return "", err
	}

	if value == nil {
		return "", nil
	}

	return fmt.Sprintf("%v", value), nil
}

// artifactText renders a loaded artifact for an instruction. A text artifact
// can come back as inline data rather than as Text: gcsartifact stores a text
// Part as a text/plain object and loads it back as bytes. Inline data goes
// through the checks of adk-python's as_safe_part_for_llm
// (load_artifacts_tool.py), which its _artifact_to_text calls, in the same
// order:
//
//  1. Inline data whose MIME type Gemini accepts inline (image/*, audio/*,
//     video/*, application/pdf, apart from the SVG and XML image types) is
//     never decoded, whatever the artifact's name.
//  2. Otherwise it is decoded when its MIME type is text/* or one of
//     adk-python's _TEXT_LIKE_MIME_TYPES, or its name has a .csv, .txt, .json
//     or .xml extension.
//
// Where this differs from adk-python:
//   - Inline data of the first kind renders as an empty string, as before.
//     adk-python's _artifact_to_text falls back to str(part) there, which
//     writes the Part's field dump, raw bytes included, into the instruction.
//   - Other binary data renders as an empty string too, where adk-python
//     renders a placeholder that describes it, and DOCX text is not
//     extracted. Neither is ported here.
//   - Invalid UTF-8 becomes one U+FFFD per run of invalid bytes, as
//     strings.ToValidUTF8 does, where Python's errors='replace' can give one
//     per byte.
func artifactText(name string, part *genai.Part) string {
	if part.InlineData == nil {
		return part.Text
	}
	mimeType := normalizeMIMEType(part.InlineData.MIMEType)
	if isGeminiInlineMIMEType(mimeType) {
		return part.Text
	}
	if isTextLikeMIMEType(mimeType) || hasTextFileExtension(name) {
		return strings.ToValidUTF8(string(part.InlineData.Data), "\uFFFD")
	}
	return part.Text
}

// normalizeMIMEType returns the lowercase media type without parameters such
// as charset.
func normalizeMIMEType(mimeType string) string {
	mediaType, _, _ := strings.Cut(mimeType, ";")
	return strings.ToLower(strings.TrimSpace(mediaType))
}

// isGeminiInlineMIMEType reports whether Gemini accepts the normalized MIME
// type as inline data, as adk-python's _is_inline_mime_type_supported does.
func isGeminiInlineMIMEType(mimeType string) bool {
	switch mimeType {
	case "image/svg", "image/svg+xml", "image/xml":
		return false
	case "application/pdf":
		return true
	}
	return strings.HasPrefix(mimeType, "image/") || strings.HasPrefix(mimeType, "audio/") || strings.HasPrefix(mimeType, "video/")
}

// isTextLikeMIMEType reports whether the normalized MIME type is one
// adk-python decodes as text.
func isTextLikeMIMEType(mimeType string) bool {
	if strings.HasPrefix(mimeType, "text/") {
		return true
	}
	switch mimeType {
	case "application/csv", "application/json", "application/svg+xml", "application/xml",
		"image/svg", "image/svg+xml", "image/xml":
		return true
	}
	return false
}

func hasTextFileExtension(name string) bool {
	switch strings.ToLower(path.Ext(name)) {
	case ".csv", ".txt", ".json", ".xml":
		return true
	}
	return false
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

		// Leave literal placeholders in place; they are copied with the
		// surrounding text.
		if startIndex > 0 && strings.IndexByte(literalPlaceholderPrefixes, template[startIndex-1]) >= 0 {
			continue
		}

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
