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
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/typeutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

// unnamedTool is the name the flow answers with for a function call that has
// no name, as adk-python's caller does (flows/llm_flows/tools/_caller.py:453).
const unnamedTool = "<unnamed>"

// toolResult is the final result of one tool call, the value callbacks saw
// and returned, together with what the flow needs to search it for media the
// way adk-python searches the raw value (_caller.py:331-335).
type toolResult struct {
	// m is the result as a map; nil when the tool returned nothing (a nil
	// map from Run, or nil from RunRaw).
	m map[string]any
	// wrapper is the map normalizeRawResult made for a non-map raw value, and
	// raw is that value. Both are unset for a map result.
	wrapper map[string]any
	raw     any
}

// normalizeRawResult is _normalize_tool_result (_caller.py:66-70) for the
// value callbacks see: a map[string]any is used as it is, so a callback that
// edits it in place edits the result, nil stays a nil map, which the flow
// wraps after the callbacks, and any other value v becomes a new map
// {"result": v}, remembered with v.
func normalizeRawResult(raw any) toolResult {
	if raw == nil {
		return toolResult{}
	}
	if m, ok := raw.(map[string]any); ok {
		return toolResult{m: m}
	}
	w := map[string]any{"result": raw}
	return toolResult{m: w, wrapper: w, raw: raw}
}

// wrapsRaw reports whether r.m is still the flow's wrapper holding the raw
// value alone: the same map, with the one key "result", whose value equals
// r.raw. A callback that adds a key or replaces "result" in place ends it, as
// a Python callback returning a new dict would.
func (r toolResult) wrapsRaw() bool {
	if r.wrapper == nil || len(r.m) != 1 || reflect.ValueOf(r.m).UnsafePointer() != reflect.ValueOf(r.wrapper).UnsafePointer() {
		return false
	}
	v, ok := r.m["result"]
	return ok && reflect.DeepEqual(v, r.raw)
}

// buildFunctionResponse is _build_function_response_content (_caller.py:271-317)
// together with the extraction and normalization of _build_response_event
// (_caller.py:331-335). It moves the media of res into the response's Parts,
// searching the raw value when res still wraps it, wraps a remainder that is
// not a map as {"result": remainder}, and sets Scheduling to override, or, when
// override is "", to t's tool.ResponseScheduler value.
//
// t may be nil, which gives no scheduling; name is the name to answer with.
// res.m may be nil, which gives an empty response. The part shares res.m's
// values, and res is not modified.
func buildFunctionResponse(t tool.Tool, callID, name string, res toolResult, override genai.FunctionResponseScheduling) *genai.Part {
	var input any = res.m
	if res.wrapsRaw() {
		input = res.raw
	}
	remaining, parts := toolinternal.ExtractMedia(input)
	response, ok := remaining.(map[string]any)
	if !ok {
		response = map[string]any{"result": remaining}
	}
	fr := &genai.FunctionResponse{ID: callID, Name: name, Response: response, Parts: parts}
	fr.Scheduling = override
	if s, ok := t.(tool.ResponseScheduler); ok && override == "" {
		fr.Scheduling = s.ResponseScheduling()
	}
	return &genai.Part{FunctionResponse: fr}
}

// isFalsyResult reports whether res is a result Python treats as false: an
// empty or nil map, or the flow's wrapper of a raw value that is nil, false,
// zero, an empty string or an empty slice, array or map. A long-running or
// deferring tool's call with such a result gets no function response
// (_caller.py:618-628).
func isFalsyResult(res toolResult) bool {
	if len(res.m) == 0 {
		return true
	}
	return res.wrapsRaw() && falsy(res.raw)
}

func falsy(v any) bool {
	if v == nil {
		return true
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Bool:
		return !rv.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return rv.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return rv.Float() == 0
	case reflect.Complex64, reflect.Complex128:
		return rv.Complex() == 0
	case reflect.String, reflect.Slice, reflect.Array, reflect.Map:
		return rv.Len() == 0
	case reflect.Pointer, reflect.Interface, reflect.Func, reflect.Chan:
		return rv.IsNil()
	}
	return false
}

// toolLookupFailure is a function call whose name no runnable tool has.
type toolLookupFailure struct {
	name  string   // the called name, or unnamedTool
	order []string // the callable names in registry order
	err   error    // the error on-tool-error callbacks receive, made once per call
}

func newToolLookupFailure(name string, order []string) *toolLookupFailure {
	return &toolLookupFailure{name: name, order: order, err: errors.New(toolNotFoundText(name, order))}
}

// response is the answer the model gets when no callback answers the call.
func (l *toolLookupFailure) response() map[string]any {
	return toolNotFoundResponse(l.name, l.order)
}

// toolNotFoundText is the error text on-tool-error callbacks receive for an
// unknown name: adk-python's _get_tool text (_caller.py:90-98), starting with
// a lower-case "tool" as a Go error does.
func toolNotFoundText(name string, order []string) string {
	return fmt.Sprintf(`tool '%s' not found.
Available tools: %s

Possible causes:
  1. LLM hallucinated the function name - review agent instruction clarity
  2. Tool not registered - verify agent.tools list
  3. Name mismatch - check for typos

Suggested fixes:
  - Review agent instruction to ensure tool usage is clear
  - Verify tool is included in agent.tools list
  - Check for typos in function name`, name, strings.Join(order, ", "))
}

// toolNotFoundResponse is the answer the model gets for an unknown name,
// adk-python's build_tool_not_found_response (_error_handler.py:88-95).
func toolNotFoundResponse(name string, order []string) map[string]any {
	available := strings.Join(order, ", ")
	if available == "" {
		available = "none"
	}
	return map[string]any{"error": fmt.Sprintf("Invoking `%s()` failed as no tool with that name is available. The tools you can call are: %s. You could retry, but it is IMPORTANT that you only call a tool from that list.", name, available)}
}

// registryOrder returns each key of tools once, in the order the tools
// entered the request, as adk-python's tool dict keeps insertion order: the
// names of req's function declarations, then the names of fTools not yet
// listed, then any other key, sorted. req and fTools may be nil; it returns
// nil for no tools.
func registryOrder(req *model.LLMRequest, fTools []tool.Tool, tools map[string]tool.Tool) []string {
	if len(tools) == 0 {
		return nil
	}
	order := make([]string, 0, len(tools))
	listed := make(map[string]bool, len(tools))
	add := func(name string) {
		if _, ok := tools[name]; ok && !listed[name] {
			listed[name] = true
			order = append(order, name)
		}
	}
	if req != nil && req.Config != nil {
		for _, t := range req.Config.Tools {
			if t == nil {
				continue
			}
			for _, d := range t.FunctionDeclarations {
				if d != nil {
					add(d.Name)
				}
			}
		}
	}
	for _, t := range fTools {
		add(t.Name())
	}
	var rest []string
	for name := range tools {
		if !listed[name] {
			rest = append(rest, name)
		}
	}
	slices.Sort(rest)
	return append(order, rest...)
}

// cloneArgs returns a deep copy of a call's arguments, or an empty map when
// it has none, as adk-python's caller does (_caller.py:441-443), so that a
// callback or tool cannot change the model's function call.
func cloneArgs(m map[string]any) map[string]any {
	if len(m) == 0 {
		return map[string]any{}
	}
	return typeutil.DeepCopy(m).(map[string]any)
}

// runTool runs t, a toolinternal.FunctionTool, through RunRaw when it has one,
// so that a raw result that is not a map reaches the response builder as it
// is.
func runTool(toolCtx agent.Context, t tool.Tool, args map[string]any) (toolResult, error) {
	if r, ok := t.(toolinternal.RawResultTool); ok {
		raw, err := r.RunRaw(toolCtx, args)
		if err != nil {
			return toolResult{}, err
		}
		return normalizeRawResult(raw), nil
	}
	m, err := t.(toolinternal.FunctionTool).Run(toolCtx, args)
	return toolResult{m: m}, err
}

// answerLookupFailure runs the on-tool-error callbacks for a call no tool and
// no before-tool callback answered. A callback's map answers it; a callback
// error other than lookup.err becomes the error response; otherwise the model
// gets the not-found response.
func (f *Flow) answerLookupFailure(toolCtx agent.Context, t tool.Tool, args map[string]any, lookup *toolLookupFailure) toolResult {
	response, err := f.runOnToolErrorCallbacks(toolCtx, t, args, lookup.err)
	if response != nil {
		return toolResult{m: response}
	}
	if err != nil && !errors.Is(err, lookup.err) {
		errText, _ := toolinternal.ModelErrorText(lookup.name, err)
		return toolResult{m: map[string]any{"error": errText}}
	}
	return toolResult{m: lookup.response()}
}
