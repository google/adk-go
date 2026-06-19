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
	"iter"
	"regexp"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/codeexecution"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func codeExecutionRequestProcessor(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		llmAgent := asLLMAgent(ctx.Agent())
		if llmAgent == nil {
			return
		}
		codeExecutor := Reveal(llmAgent).CodeExecutor
		if codeExecutor == nil {
			return
		}

		// For the BuiltInCodeExecutor only, code execution is handled by the model itself.
		if codeExecutorProcessor, ok := codeExecutor.(toolinternal.RequestProcessor); ok {
			toolCtx := agent.NewToolContext(ctx, "", &session.EventActions{}, nil)
			if err := codeExecutorProcessor.ProcessRequest(toolCtx, req); err != nil {
				yield(nil, err)
				return
			}
		}

		// TODO: consider input file processing implementation (see adk-python src/google/adk/flows/llm_flows/_code_execution.py)
		// This involves extracting any data files from the session history, executing analysis code and storing them in memory.
	}
}

func codeExecutionResponseProcessor(ctx agent.InvocationContext, req *model.LLMRequest, resp *model.LLMResponse) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		llmAgent := asLLMAgent(ctx.Agent())
		if llmAgent == nil {
			return
		}
		if resp == nil || resp.Content == nil || resp.Partial {
			return // Skip if the response is empty or partial
		}
		codeExecutor := Reveal(llmAgent).CodeExecutor
		if codeExecutor == nil {
			return // Skip if no configured CodeExecutor
		}

		// TODO: consider generated image handling (see adk-python src/google/adk/flows/llm_flows/_code_execution.py)

		codeExecutorContext := codeexecution.NewCodeExecutorContext(ctx.Session().State())

		// Skip if the error count exceeds the max retry attempts.
		if codeExecutorContext.GetErrorCount(ctx.InvocationID()) > codeExecutor.Config().ErrorRetryAttempts {
			return
		}

		// Extract the code from model response and truncate the content
		// to the part with the first code block. Do nothing if no code to execute.
		codeStr := extractCodeAndTruncateContent(resp.Content, codeExecutor.Config().CodeBlockDelimiters)
		if codeStr == "" {
			return
		}

		// Emit code execution event
		ev := session.NewEventWithContext(ctx, ctx.InvocationID())
		ev.LLMResponse = model.LLMResponse{
			Content: resp.Content,
		}
		ev.Author = ctx.Agent().Name()
		ev.Branch = ctx.Branch()
		if !yield(ev, nil) {
			return
		}

		// Execute the code
		executionResult, err := codeExecutor.ExecuteCode(ctx, &codeexecution.CodeExecutionInput{
			Code:        codeStr,
			ExecutionId: getOrSetExecutionId(ctx, codeExecutorContext),
			InputFiles:  codeExecutorContext.GetInputFiles(),
		})
		if err != nil {
			yield(nil, err)
			return
		}
		if err = codeExecutorContext.UpdateCodeExecutionResult(ctx.InvocationID(), codeStr, executionResult.Stdout, executionResult.Stderr); err != nil {
			yield(nil, err)
			return
		}

		// Emit code execution result event
		ev, err = generateCodeExecutionResultEvent(ctx, codeExecutorContext, executionResult)
		if !yield(ev, err) {
			return
		}

		// Skip processing the original model response to continue generation loop.
		resp.Content = nil
	}
}

// generateCodeExecutionResultEvent constructs a code execution result event.
func generateCodeExecutionResultEvent(ctx agent.InvocationContext, codeExecutionContext *codeexecution.CodeExecutorContext, codeExecutionResult *codeexecution.CodeExecutionResult) (*session.Event, error) {
	ev := session.NewEventWithContext(ctx, ctx.InvocationID())
	ev.LLMResponse = model.LLMResponse{
		Content: &genai.Content{
			Role: genai.RoleModel,
			Parts: []*genai.Part{
				buildCodeExecutionResultPart(codeExecutionResult),
			},
		},
	}
	ev.Author = ctx.Agent().Name()
	ev.Branch = ctx.Branch()
	ev.Actions.StateDelta = codeExecutionContext.GetStateDelta()

	// Handle code execution error retry.
	if codeExecutionResult.Stderr != "" {
		if err := codeExecutionContext.IncrementErrorCount(ctx.InvocationID()); err != nil {
			return nil, err
		}
	} else {
		if err := codeExecutionContext.ResetErrorCount(ctx.InvocationID()); err != nil {
			return nil, err
		}
	}

	// Handle output files.
	for _, outputFile := range codeExecutionResult.OutputFiles {
		saveResp, err := ctx.Artifacts().Save(ctx, outputFile.Name, genai.NewPartFromBytes(outputFile.Content, outputFile.MimeType))
		if err != nil {
			return nil, err
		}
		ev.Actions.ArtifactDelta[outputFile.Name] = saveResp.Version
	}
	return ev, nil
}

func buildExecutableCodePart(code string) *genai.Part {
	return genai.NewPartFromExecutableCode(code, genai.LanguagePython)
}

func buildCodeExecutionResultPart(code *codeexecution.CodeExecutionResult) *genai.Part {
	if code.Stderr != "" {
		return genai.NewPartFromCodeExecutionResult(
			genai.OutcomeFailed, code.Stderr,
		)
	}
	return genai.NewPartFromCodeExecutionResult(
		genai.OutcomeOK, code.Stdout,
	)
}

// extractCodeAndTruncateContent extracts the first code block from the content and truncate everything after it.
func extractCodeAndTruncateContent(content *genai.Content, delimiters []codeexecution.Delimiter) string {
	if content == nil || len(content.Parts) == 0 {
		return ""
	}

	// TODO: Extract the code from the executable code parts if there are no associated
	// code execution result parts.

	// Extract from text parts
	var textParts []*genai.Part
	for _, part := range content.Parts {
		if part.Text != "" {
			textParts = append(textParts, part)
		}
	}
	if len(textParts) == 0 {
		return ""
	}
	var responseTexts []string
	for _, part := range textParts {
		responseTexts = append(responseTexts, part.Text)
	}
	responseText := strings.Join(responseTexts, "\n")

	var leadingPatterns []string
	var trailingPatterns []string
	for _, d := range delimiters {
		leadingPatterns = append(leadingPatterns, d.Leading)
		trailingPatterns = append(trailingPatterns, d.Trailing)
	}
	leadingDelimiterPattern := strings.Join(leadingPatterns, "|")
	trailingDelimiterPattern := strings.Join(trailingPatterns, "|")

	patternStr := fmt.Sprintf("(?s)%s(.*?)%s",
		regexp.QuoteMeta(leadingDelimiterPattern), regexp.QuoteMeta(trailingDelimiterPattern))
	pattern := regexp.MustCompile(patternStr)
	match := pattern.FindStringSubmatch(responseText)
	if len(match) < 2 {
		return "" // No match found
	}
	codeStr := match[1]
	if codeStr == "" {
		return ""
	}
	content.Parts = []*genai.Part{}
	content.Parts = append(
		content.Parts,
		buildExecutableCodePart(codeStr),
	)
	return codeStr
}

func getOrSetExecutionId(ctx agent.InvocationContext, codeExecutorContext *codeexecution.CodeExecutorContext) string {
	executionId := codeExecutorContext.GetExecutionId()
	if executionId == "" {
		executionId = ctx.Session().ID()
	}
	codeExecutorContext.SetExecutionId(executionId)
	return executionId
}
