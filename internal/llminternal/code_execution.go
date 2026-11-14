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

package llminternal

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/codeexecutor"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/platform"
	"google.golang.org/adk/v2/session"
)

// artifactServiceMissingText is the message of the error a code execution
// returns when it needs the artifact service and the invocation has none. It
// is adk-go's wording for that condition (agent/common_context.go).
const artifactServiceMissingText = "artifact service is not initialized"

// codeExecutionPostProcess runs the first code block of resp through the
// agent's code executor (adk-python flows/llm_flows/extensions/
// _code_execution.py, _run_post_processor).
//
// It yields nothing and leaves resp untouched for a nil or partial resp, a nil
// resp.Content, an agent without an executor, a closed retry gate, and a
// response without code. For a codeexecutor.BuiltIn it saves the response's
// images as artifacts, rewrites their parts in place and yields one event
// without content. For any other executor it yields the code event, runs the
// code, yields the result event and then sets resp.Content to nil, so the flow
// skips the model event and calls the model again. It stops at once when yield
// returns false, and on a failure it yields the error and returns, leaving
// resp.Content as it was.
func codeExecutionPostProcess(ctx agent.InvocationContext, resp *model.LLMResponse) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if resp == nil || resp.Partial || resp.Content == nil {
			return
		}
		exec := getCodeExecutor(ctx)
		if exec == nil {
			return
		}
		if _, ok := exec.(codeexecutor.BuiltIn); ok {
			ev, err := builtInImageEvent(ctx, resp)
			if err != nil {
				yield(nil, err)
				return
			}
			yield(ev, nil)
			return
		}
		settings := exec.Settings().WithDefaults()
		ec, err := newCodeExecutorContext(ctx.Session().State())
		if err != nil {
			yield(nil, fmt.Errorf("prepare code execution: %w", err))
			return
		}
		errorCount, err := ec.errorCount(ctx.InvocationID())
		if err != nil {
			yield(nil, fmt.Errorf("prepare code execution: %w", err))
			return
		}
		if errorCount >= *settings.ErrorRetryAttempts {
			return
		}
		kept, thoughts := splitThoughts(resp.Content.Parts)
		code, eventParts, ok := extractCode(kept, settings.CodeBlockDelimiters)
		if !ok {
			return
		}
		// The thoughts go back in front: the model expects its own signatures
		// back in history.
		codeEvent := newExecutionEvent(ctx, &genai.Content{
			Role:  resp.Content.Role,
			Parts: append(slices.Clone(thoughts), eventParts...),
		})
		if !yield(codeEvent, nil) {
			return
		}
		res, err := runCode(ctx, exec, settings, ec, code)
		if err != nil {
			yield(nil, err)
			return
		}
		resultEvent, err := postProcessExecutionResult(ctx, ec, res)
		if err != nil {
			yield(nil, err)
			return
		}
		if !yield(resultEvent, nil) {
			return
		}
		resp.Content = nil
	}
}

// splitThoughts returns the non-nil parts of parts in two new slices: the
// thoughts, which are the parts with Thought set, and the rest. Both keep the
// order of parts and share its *genai.Part values; parts is not modified.
func splitThoughts(parts []*genai.Part) (kept, thoughts []*genai.Part) {
	for _, p := range parts {
		switch {
		case p == nil:
		case p.Thought:
			thoughts = append(thoughts, p)
		default:
			kept = append(kept, p)
		}
	}
	return kept, thoughts
}

// extractCode finds the first code block of parts, which hold no nil part,
// and returns its code and the parts of the code event (adk-python
// code_executors/code_execution_utils.py, extract_code_and_truncate_content).
//
// The first executable-code part not followed by a code execution result
// decides: its code, and the parts up to and including it. Otherwise the text
// parts are joined with "\n" and searched for each (opening, closing) pair of
// delimiters; the block whose opening starts earliest wins, and the event
// parts are the text before it (when not empty, in a copy of the first text
// part) and an executable-code part. ok is false when no block is found or
// its code is empty. extractCode never modifies parts or the parts they point
// to.
func extractCode(parts []*genai.Part, delimiters [][2]string) (code string, eventParts []*genai.Part, ok bool) {
	for i, p := range parts {
		if p.ExecutableCode == nil {
			continue
		}
		if i < len(parts)-1 && parts[i+1].CodeExecutionResult != nil {
			continue
		}
		code = p.ExecutableCode.Code
		if code == "" {
			return "", nil, false
		}
		return code, slices.Clone(parts[:i+1]), true
	}

	var firstText *genai.Part
	var texts []string
	for _, p := range parts {
		if p.Text == "" {
			continue
		}
		if firstText == nil {
			firstText = p
		}
		texts = append(texts, p.Text)
	}
	if firstText == nil {
		return "", nil, false
	}
	text := strings.Join(texts, "\n")

	bestStart, bestEnd, bestLeadLen := -1, -1, 0
	for _, d := range delimiters {
		start := strings.Index(text, d[0])
		if start == -1 {
			continue
		}
		codeStart := start + len(d[0])
		end := strings.Index(text[codeStart:], d[1])
		if end == -1 {
			continue
		}
		if bestStart == -1 || start < bestStart {
			bestStart, bestEnd, bestLeadLen = start, codeStart+end, len(d[0])
		}
	}
	if bestStart == -1 {
		return "", nil, false
	}
	code = text[bestStart+bestLeadLen : bestEnd]
	if code == "" {
		return "", nil, false
	}
	if prefix := text[:bestStart]; prefix != "" {
		prefixPart := clone(firstText)
		prefixPart.Text = prefix
		eventParts = append(eventParts, prefixPart)
	}
	eventParts = append(eventParts, genai.NewPartFromExecutableCode(code, genai.LanguagePython))
	return code, eventParts, true
}

// runCode runs code, which is not empty, through exec and records the
// execution in ec. settings are exec's, with defaults applied. The executor
// gets the stored input files and, when settings.Stateful is true, the
// execution ID; ctx is both its context and its invocation context.
//
// An error of the executor is returned wrapped, and a nil Result with a nil
// error is the error "execute code: code executor returned no result"; in both
// cases no execution is recorded. No message names the code or its output.
func runCode(ctx agent.InvocationContext, exec codeexecutor.CodeExecutor, settings codeexecutor.Settings, ec *codeExecutorContext, code string) (*codeexecutor.Result, error) {
	files, err := ec.inputFiles()
	if err != nil {
		return nil, fmt.Errorf("prepare code execution: %w", err)
	}
	id, err := ec.getOrSetExecutionID(ctx.Session().ID(), settings.Stateful)
	if err != nil {
		return nil, fmt.Errorf("prepare code execution: %w", err)
	}
	res, err := exec.ExecuteCode(ctx, ctx, &codeexecutor.Input{Code: code, InputFiles: files, ExecutionID: id})
	if err != nil {
		return nil, fmt.Errorf("execute code: %w", err)
	}
	if res == nil {
		return nil, errors.New("execute code: code executor returned no result")
	}
	if err := ec.recordExecution(ctx.InvocationID(), code, res.Stdout, res.Stderr, platform.Now(ctx)); err != nil {
		return nil, fmt.Errorf("record code execution: %w", err)
	}
	return res, nil
}

// postProcessExecutionResult returns the result event of res, a non-nil
// Result (adk-python _post_process_code_execution_result). The event holds one
// code execution result part with role model, the code execution context as
// its state delta, and the version of each output file, which it saves as an
// artifact, in its artifact delta. A non-empty res.Stderr adds one to the
// invocation's error count; otherwise the count is reset.
//
// Without an artifact service it fails with "artifact service is not
// initialized", also when res has no output files. Steps done before a failure
// stay done: the count change survives a failed save, and files saved before
// it stay saved.
func postProcessExecutionResult(ctx agent.InvocationContext, ec *codeExecutorContext, res *codeexecutor.Result) (*session.Event, error) {
	if artifactServiceMissing(ctx.Artifacts()) {
		return nil, errors.New(artifactServiceMissingText)
	}
	for i, f := range res.OutputFiles {
		if f == nil {
			return nil, fmt.Errorf("output file %d is nil", i)
		}
	}
	part := buildCodeExecutionResultPart(res)
	delta, err := ec.stateDelta()
	if err != nil {
		return nil, fmt.Errorf("read code execution context: %w", err)
	}
	if res.Stderr != "" {
		err = ec.incrementErrorCount(ctx.InvocationID())
	} else {
		err = ec.resetErrorCount(ctx.InvocationID())
	}
	if err != nil {
		return nil, fmt.Errorf("update code execution error count: %w", err)
	}
	ev := newExecutionEvent(ctx, &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{part}})
	ev.Actions.StateDelta = delta
	for i, f := range res.OutputFiles {
		mimeType := f.MimeType
		if mimeType == "" {
			mimeType = "text/plain"
		}
		saved, err := ctx.Artifacts().Save(ctx, f.Name, genai.NewPartFromBytes(f.Content, mimeType))
		if err != nil {
			return nil, fmt.Errorf("save output file %d: %w", i, err)
		}
		ev.Actions.ArtifactDelta[f.Name] = saved.Version
	}
	return ev, nil
}

// buildCodeExecutionResultPart returns the code execution result part of res,
// a non-nil Result whose OutputFiles hold no nil element: OUTCOME_FAILED with
// the stderr when res.Stderr is not empty, else OUTCOME_OK with the stdout
// and the names of the output files (adk-python code_execution_utils.py,
// build_code_execution_result_part).
func buildCodeExecutionResultPart(res *codeexecutor.Result) *genai.Part {
	if res.Stderr != "" {
		return genai.NewPartFromCodeExecutionResult(genai.OutcomeFailed, res.Stderr)
	}
	var pieces []string
	if res.Stdout != "" || len(res.OutputFiles) == 0 {
		pieces = append(pieces, "Code execution result:\n"+res.Stdout+"\n")
	}
	if len(res.OutputFiles) > 0 {
		names := make([]string, len(res.OutputFiles))
		for i, f := range res.OutputFiles {
			names[i] = "`" + f.Name + "`"
		}
		pieces = append(pieces, "Saved artifacts:\n"+strings.Join(names, ","))
	}
	return genai.NewPartFromCodeExecutionResult(genai.OutcomeOK, strings.Join(pieces, "\n\n"))
}

// builtInImageEvent saves every image part of resp, whose Content is not nil,
// as an artifact and returns the event that names them, which has no content
// (adk-python _run_post_processor, the BuiltInCodeExecutor branch). Each saved
// part is rewritten in place to the text "Saved as artifact: <name>. ". A
// response without an image still gets the event, with empty deltas.
//
// It needs the artifact service only when it finds an image, and then fails
// with "artifact service is not initialized" when there is none. Images saved
// before a failure stay saved and their parts stay rewritten.
func builtInImageEvent(ctx agent.InvocationContext, resp *model.LLMResponse) (*session.Event, error) {
	ev := newExecutionEvent(ctx, nil)
	for i, p := range resp.Content.Parts {
		if p == nil || p.InlineData == nil || !strings.HasPrefix(p.InlineData.MIMEType, "image/") {
			continue
		}
		if artifactServiceMissing(ctx.Artifacts()) {
			return nil, errors.New(artifactServiceMissingText)
		}
		blob := p.InlineData
		name := imageArtifactName(ctx, blob)
		if blob.Data == nil {
			return nil, fmt.Errorf("built-in code execution: image part %d has no data", i)
		}
		saved, err := ctx.Artifacts().Save(ctx, name, genai.NewPartFromBytes(blob.Data, blob.MIMEType))
		if err != nil {
			return nil, fmt.Errorf("save image artifact %d: %w", i, err)
		}
		ev.Actions.ArtifactDelta[name] = saved.Version
		p.InlineData = nil
		p.Text = "Saved as artifact: " + name + ". "
	}
	return ev, nil
}

// imageArtifactName returns the artifact name of blob, an image: its display
// name, or else the time ctx's clock gives, in the host's time zone, as
// YYYYMMDD_HHMMSS followed by "." and the MIME subtype. Two unnamed images of
// one subtype in one second get the same name, as in adk-python.
func imageArtifactName(ctx context.Context, blob *genai.Blob) string {
	if blob.DisplayName != "" {
		return blob.DisplayName
	}
	ext := blob.MIMEType[strings.LastIndex(blob.MIMEType, "/")+1:]
	return platform.Now(ctx).In(time.Local).Format("20060102_150405") + "." + ext
}

// newExecutionEvent returns a new event of the agent in ctx, with ctx's
// invocation ID and branch, empty deltas and content, which it stores as given
// (nil allowed). The caller owns content's Role: an event whose content has no
// role is left out of later model requests.
func newExecutionEvent(ctx agent.InvocationContext, content *genai.Content) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = ctx.Agent().Name()
	ev.Branch = ctx.Branch()
	ev.Content = content
	return ev
}

// artifactServiceMissing reports whether arts stands for an invocation without
// an artifact service: arts is nil, or it is the stand-in an agent.Context
// returns then, which reports the missing service through
// ArtifactServiceErr.
func artifactServiceMissing(arts agent.Artifacts) bool {
	if arts == nil {
		return true
	}
	_, missing := arts.(interface{ ArtifactServiceErr() error })
	return missing
}

// convertCodeExecutionHistory replaces each entry of contents that ends in
// executable code or consists of one code execution result with a copy that
// holds the code or the result as fenced text (adk-python
// _CodeExecutionRequestProcessor). s are the executor's settings as it returns
// them; the code is fenced with the first pair of code block delimiters, or
// with nothing when there is none, and the result with the execution result
// delimiters. It never modifies a Content or a Part.
func convertCodeExecutionHistory(contents []*genai.Content, s codeexecutor.Settings) {
	s = s.WithDefaults()
	var block [2]string
	if len(s.CodeBlockDelimiters) > 0 {
		block = s.CodeBlockDelimiters[0]
	}
	result := *s.ExecutionResultDelimiters
	for i, c := range contents {
		contents[i] = convertCodeExecutionParts(c, block, result)
	}
}

// convertCodeExecutionParts returns c with its code execution parts as text
// (adk-python code_execution_utils.py, convert_code_execution_parts): a last
// part holding executable code becomes block[0] + code + block[1]; a content
// whose only part is a code execution result becomes a user content with the
// text result[0] + output + result[1], or "" when the output is empty. It
// returns c itself when nothing changes, and a new Content otherwise.
func convertCodeExecutionParts(c *genai.Content, block, result [2]string) *genai.Content {
	if c == nil || len(c.Parts) == 0 {
		return c
	}
	last := c.Parts[len(c.Parts)-1]
	switch {
	case last == nil:
		return c
	case last.ExecutableCode != nil:
		parts := slices.Clone(c.Parts)
		parts[len(parts)-1] = &genai.Part{Text: block[0] + last.ExecutableCode.Code + block[1]}
		return &genai.Content{Role: c.Role, Parts: parts}
	case len(c.Parts) == 1 && last.CodeExecutionResult != nil:
		// genai's Output is a string, so an empty output stands for
		// adk-python's unset one.
		text := ""
		if out := last.CodeExecutionResult.Output; out != "" {
			text = result[0] + out + result[1]
		}
		return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: text}}}
	}
	return c
}
